package com.example.dragonpitch.presentation

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.wear.compose.foundation.lazy.TransformingLazyColumn
import androidx.wear.compose.foundation.lazy.rememberTransformingLazyColumnState
import androidx.wear.compose.material3.*
import androidx.wear.compose.material3.lazy.rememberTransformationSpec
import androidx.wear.compose.material3.lazy.transformedHeight
import com.example.dragonpitch.presentation.theme.DragonPitchTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            DragonPitchTheme {
                AppScaffold {
                    AppNavigator()
                }
            }
        }
    }
}

// ── Navigation ──────────────────────────────────────────────────────────────
@Composable
fun AppNavigator(vm: SessionViewModel = viewModel()) {
    val uiState by vm.uiState.collectAsState()
    if (uiState.selectedPlayer == null) {
        PlayerSelectScreen(vm = vm)
    } else {
        SessionScreen(vm = vm)
    }
}

// ── Player Selection Screen ──────────────────────────────────────────────────
@Composable
fun PlayerSelectScreen(vm: SessionViewModel) {
    val players = listOf("Alex", "Jordan", "Priya")
    val listState = rememberTransformingLazyColumnState()
    val transformationSpec = rememberTransformationSpec()
    ScreenScaffold(scrollState = listState) { contentPadding ->
        TransformingLazyColumn(
            contentPadding = contentPadding,
            state = listState
        ) {
            item {
                ListHeader(
                    modifier = Modifier
                        .fillMaxWidth()
                        .transformedHeight(this, transformationSpec),
                    transformation = SurfaceTransformation(transformationSpec),
                ) {
                    Text("🐉 Who is pitching?")
                }
            }
            items(players.size) { index ->
                val player = players[index]
                Button(
                    onClick = { vm.selectPlayer(player) },
                    modifier = Modifier
                        .fillMaxWidth()
                        .transformedHeight(this, transformationSpec),
                    transformation = SurfaceTransformation(transformationSpec),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = MaterialTheme.colorScheme.primary
                    )
                ) {
                    Text(player)
                }
            }
        }
    }
}

// ── Session Screen ───────────────────────────────────────────────────────────
@Composable
fun SessionScreen(vm: SessionViewModel) {
    val uiState by vm.uiState.collectAsState()
    val listState = rememberTransformingLazyColumnState()
    val transformationSpec = rememberTransformationSpec()
    ScreenScaffold(scrollState = listState) { contentPadding ->
        TransformingLazyColumn(
            contentPadding = contentPadding,
            state = listState
        ) {
            item {
                ListHeader(
                    modifier = Modifier
                        .fillMaxWidth()
                        .transformedHeight(this, transformationSpec),
                    transformation = SurfaceTransformation(transformationSpec),
                ) {
                    Text("🐉 ${uiState.selectedPlayer}")
                }
            }
            item {
                Text(
                    text = uiState.statusMessage,
                    modifier = Modifier
                        .fillMaxWidth()
                        .transformedHeight(this, transformationSpec),
                )
            }
            if (uiState.isRunning) {
                item {
                    Text(
                        text = "Pitches: ${uiState.pitchCount}",
                        modifier = Modifier
                            .fillMaxWidth()
                            .transformedHeight(this, transformationSpec),
                    )
                }
                if (uiState.latestVerdictText.isNotEmpty()) {
                    item {
                        Text(
                            text = uiState.latestVerdictText,
                            modifier = Modifier
                                .fillMaxWidth()
                                .transformedHeight(this, transformationSpec),
                        )
                    }
                }
            }
            item {
                Button(
                    onClick = { vm.startSession() },
                    enabled = !uiState.isRunning,
                    modifier = Modifier
                        .fillMaxWidth()
                        .transformedHeight(this, transformationSpec),
                    transformation = SurfaceTransformation(transformationSpec),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = MaterialTheme.colorScheme.primary
                    )
                ) {
                    Text("▶ Start Session")
                }
            }
            item {
                Button(
                    onClick = { vm.stopSession() },
                    enabled = uiState.isRunning,
                    modifier = Modifier
                        .fillMaxWidth()
                        .transformedHeight(this, transformationSpec),
                    transformation = SurfaceTransformation(transformationSpec),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = MaterialTheme.colorScheme.error
                    )
                ) {
                    Text("⏹ Stop Session")
                }
            }
            item {
                Button(
                    onClick = { vm.clearPlayer() },
                    enabled = !uiState.isRunning,
                    modifier = Modifier
                        .fillMaxWidth()
                        .transformedHeight(this, transformationSpec),
                    transformation = SurfaceTransformation(transformationSpec),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = MaterialTheme.colorScheme.secondary
                    )
                ) {
                    Text("↩ Change Player")
                }
            }
        }
    }
}
