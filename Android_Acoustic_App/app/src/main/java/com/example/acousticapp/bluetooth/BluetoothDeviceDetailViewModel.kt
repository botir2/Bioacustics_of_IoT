package com.example.acousticapp.bluetooth

import androidx.lifecycle.ViewModel
import com.example.acousticapp.model.BluetoothConnectionState
import com.example.acousticapp.model.BluetoothDeviceUiModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

data class BluetoothDetailState(
    val device: BluetoothDeviceUiModel? = null,
    val connectionStatus: BluetoothConnectionState = BluetoothConnectionState.NOT_CONNECTED,
    val battery: String = "Coming soon",
    val acousticStatus: String = "Coming soon",
    val lastDetection: String = "Coming soon",
    val liveAcousticData: String = "Coming soon"
)

class BluetoothDeviceDetailViewModel : ViewModel() {

    private val _uiState = MutableStateFlow(BluetoothDetailState())
    val uiState: StateFlow<BluetoothDetailState> = _uiState.asStateFlow()

    fun setDevice(device: BluetoothDeviceUiModel) {
        _uiState.value = _uiState.value.copy(
            device = device,
            connectionStatus = device.connectionState
        )
    }

    fun connect() {
        // TODO: Connect to real Bluetooth device
        _uiState.value = _uiState.value.copy(connectionStatus = BluetoothConnectionState.CONNECTING)
        // Mock success
        _uiState.value = _uiState.value.copy(connectionStatus = BluetoothConnectionState.CONNECTED)
    }

    fun disconnect() {
        // TODO: Disconnect from real Bluetooth device
        _uiState.value = _uiState.value.copy(connectionStatus = BluetoothConnectionState.DISCONNECTED)
    }
}
