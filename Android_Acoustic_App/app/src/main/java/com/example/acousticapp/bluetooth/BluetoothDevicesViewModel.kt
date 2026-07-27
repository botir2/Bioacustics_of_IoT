package com.example.acousticapp.bluetooth

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.acousticapp.model.BluetoothConnectionState
import com.example.acousticapp.model.BluetoothDeviceUiModel
import com.example.acousticapp.repository.BluetoothRepository
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

sealed class BluetoothUiState {
    object Idle : BluetoothUiState()
    object Scanning : BluetoothUiState()
    data class Success(val devices: List<BluetoothDeviceUiModel>) : BluetoothUiState()
    data class Error(val message: String) : BluetoothUiState()
    data class StatusUpdate(val isEnabled: Boolean) : BluetoothUiState()
}

class BluetoothDevicesViewModel(
    private val repository: BluetoothRepository
) : ViewModel() {

    private val _uiState = MutableStateFlow<BluetoothUiState>(BluetoothUiState.Idle)
    val uiState: StateFlow<BluetoothUiState> = _uiState.asStateFlow()

    private val _isBluetoothEnabled = MutableStateFlow(false)
    val isBluetoothEnabled: StateFlow<Boolean> = _isBluetoothEnabled.asStateFlow()

    private val discoveredDevices = mutableMapOf<String, BluetoothDeviceUiModel>()
    private var scanJob: Job? = null

    init {
        checkBluetoothStatus()
    }

    fun checkBluetoothStatus() {
        val enabled = repository.isBluetoothEnabled()
        _isBluetoothEnabled.value = enabled
    }

    fun startScan() {
        if (!repository.isBluetoothEnabled()) {
            _uiState.value = BluetoothUiState.Error("Bluetooth must be enabled to scan nearby devices.")
            return
        }

        scanJob?.cancel()
        discoveredDevices.clear()
        _uiState.value = BluetoothUiState.Scanning

        scanJob = viewModelScope.launch {
            repository.startScan().collect { device ->
                discoveredDevices[device.address] = device
                _uiState.value = BluetoothUiState.Success(discoveredDevices.values.toList())
            }
        }

        // Auto-stop scan after 10 seconds
        viewModelScope.launch {
            delay(10000)
            stopScan()
        }
    }

    fun stopScan() {
        scanJob?.cancel()
        if (_uiState.value is BluetoothUiState.Scanning) {
            _uiState.value = BluetoothUiState.Success(discoveredDevices.values.toList())
        }
    }

    fun connectToDevice(device: BluetoothDeviceUiModel) {
        val nativeDevice = device.nativeDevice ?: return
        
        updateDeviceState(device.address, BluetoothConnectionState.CONNECTING)

        repository.connectToDevice(nativeDevice) { newState ->
            updateDeviceState(device.address, newState)
        }
    }

    private fun updateDeviceState(address: String, newState: BluetoothConnectionState) {
        discoveredDevices[address]?.let {
            discoveredDevices[address] = it.copy(connectionState = newState)
            _uiState.value = BluetoothUiState.Success(discoveredDevices.values.toList())
        }
    }
}
