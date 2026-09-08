package com.example.dragonpitch.presentation

import android.app.Application
import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL

// ── SIMPLE CONNECTIONS MODE ──────────────────────────────────────────────────
// No on-device model. The watch just: (1) tells the EVK to start/stop,
// (2) polls the EVK for live pitch count + latest sequencing verdict,
// (3) tracks a simple peak-accelerometer "effort" value locally and reports
// it whenever the EVK's pitch count increases.

data class SessionUiState(
    val selectedPlayer: String? = null,
    val isRunning: Boolean = false,
    val statusMessage: String = "Ready",
    val pitchCount: Int = 0,
    val latestVerdictText: String = "",
)

class SessionViewModel(application: Application) : AndroidViewModel(application), SensorEventListener {

    // ── CONFIG — update this line whenever the EVK's IP changes ─────────────
    private val evkBaseUrl = "http://YOUR_EVK_IP:5000"  // put your Dragonwing EVK's local network IP here

    private val _uiState = MutableStateFlow(SessionUiState())
    val uiState: StateFlow<SessionUiState> = _uiState

    // ── Simple effort tracking (peak |ax| since the last reported pitch) ────
    private val sensorManager =
        application.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val accelSensor = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)
    private var currentPeakEffort = 0f
    private var sessionStartTimeMs = 0L
    private var lastKnownPitchCount = 0

    private var pollJob: Job? = null

    // ── Player Selection ──────────────────────────────────────────────────────
    fun selectPlayer(player: String) {
        _uiState.value = _uiState.value.copy(
            selectedPlayer = player,
            statusMessage = "Ready — $player"
        )
    }

    fun clearPlayer() {
        _uiState.value = _uiState.value.copy(
            selectedPlayer = null,
            isRunning = false,
            statusMessage = "Ready"
        )
    }

    // ── Session Control ──────────────────────────────────────────────────────
    fun startSession() {
        val player = _uiState.value.selectedPlayer ?: return

        currentPeakEffort = 0f
        lastKnownPitchCount = 0
        sessionStartTimeMs = System.currentTimeMillis()

        sensorManager.registerListener(this, accelSensor, SensorManager.SENSOR_DELAY_GAME)

        _uiState.value = _uiState.value.copy(
            isRunning = true,
            statusMessage = "Starting session...",
            pitchCount = 0,
            latestVerdictText = ""
        )

        viewModelScope.launch(Dispatchers.IO) {
            val ok = postJson("/session/start", JSONObject().apply { put("player", player) })
            if (!ok) {
                _uiState.value = _uiState.value.copy(
                    isRunning = false,
                    statusMessage = "Failed to start on EVK — check IP/wifi/EVK logs"
                )
                sensorManager.unregisterListener(this@SessionViewModel)
                return@launch
            }
            _uiState.value = _uiState.value.copy(statusMessage = "Session Active")
            startPolling()
        }
    }

    fun stopSession() {
        sensorManager.unregisterListener(this)
        pollJob?.cancel()

        _uiState.value = _uiState.value.copy(
            isRunning = false,
            statusMessage = "Session Complete ✓ (${_uiState.value.pitchCount} pitches)"
        )

        viewModelScope.launch(Dispatchers.IO) {
            postJson("/session/stop", JSONObject())
        }
    }

    // ── Polling loop: watches for new pitches + latest sequencing verdict ───
    private fun startPolling() {
        pollJob?.cancel()
        pollJob = viewModelScope.launch(Dispatchers.IO) {
            while (_uiState.value.isRunning) {
                val statusJson = getJson("/live_status")
                if (statusJson != null) {
                    val pitchCount = statusJson.optInt("pitch_count", lastKnownPitchCount)
                    val latest = statusJson.optJSONObject("latest_pitch")
                    val verdictText = if (latest != null) {
                        val seq = latest.optJSONObject("sequencing")
                        val verdict = seq?.opt("verdict")
                        when (verdict) {
                            true -> "Pitch #${latest.optInt("pitch_number")}: Great sequencing"
                            false -> "Pitch #${latest.optInt("pitch_number")}: Arm dominant"
                            else -> "Pitch #${latest.optInt("pitch_number")}: unscored"
                        }
                    } else ""

                    if (pitchCount > lastKnownPitchCount) {
                        // A new pitch was detected by the EVK since our last poll —
                        // report whatever peak effort we've accumulated since then.
                        val effort = currentPeakEffort
                        val timestampSec = (System.currentTimeMillis() - sessionStartTimeMs) / 1000.0
                        currentPeakEffort = 0f  // reset for the next pitch
                        launch(Dispatchers.IO) {
                            postJson("/watch/pitch", JSONObject().apply {
                                put("timestamp", timestampSec)
                                put("effort", effort.toDouble())
                            })
                        }
                    }
                    lastKnownPitchCount = pitchCount

                    _uiState.value = _uiState.value.copy(
                        pitchCount = pitchCount,
                        latestVerdictText = verdictText,
                        statusMessage = if (_uiState.value.isRunning) "Session Active" else _uiState.value.statusMessage
                    )
                } else {
                    _uiState.value = _uiState.value.copy(statusMessage = "EVK unreachable — retrying...")
                }
                delay(1500)
            }
        }
    }

    // ── Sensor callback — just tracks a running peak, no model/windowing ────
    override fun onSensorChanged(event: SensorEvent?) {
        event ?: return
        if (event.sensor.type == Sensor.TYPE_LINEAR_ACCELERATION) {
            val magnitude = kotlin.math.abs(event.values[0])
            if (magnitude > currentPeakEffort) {
                currentPeakEffort = magnitude
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    // ── Networking ───────────────────────────────────────────────────────────
    private fun postJson(path: String, body: JSONObject): Boolean {
        return try {
            val url = URL("$evkBaseUrl$path")
            val conn = url.openConnection() as HttpURLConnection
            conn.requestMethod = "POST"
            conn.setRequestProperty("Content-Type", "application/json")
            conn.doOutput = true
            conn.connectTimeout = 3000
            conn.readTimeout = 3000
            conn.outputStream.use { it.write(body.toString().toByteArray()) }
            val code = conn.responseCode
            conn.disconnect()
            Log.d(TAG, "POST $path -> $code")
            code in 200..299
        } catch (e: Exception) {
            Log.e(TAG, "POST $path failed: ${e.message}")
            false
        }
    }

    private fun getJson(path: String): JSONObject? {
        return try {
            val url = URL("$evkBaseUrl$path")
            val conn = url.openConnection() as HttpURLConnection
            conn.requestMethod = "GET"
            conn.connectTimeout = 3000
            conn.readTimeout = 3000
            val text = conn.inputStream.bufferedReader().use { it.readText() }
            conn.disconnect()
            JSONObject(text)
        } catch (e: Exception) {
            Log.e(TAG, "GET $path failed: ${e.message}")
            null
        }
    }

    companion object {
        private const val TAG = "SessionViewModel"
    }
}
