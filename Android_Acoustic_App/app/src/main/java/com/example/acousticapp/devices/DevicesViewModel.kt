package com.example.acousticapp.devices

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.acousticapp.model.AcousticDevice
import com.example.acousticapp.repository.DeviceRepository
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch

sealed class DevicesUiState {
    object Loading : DevicesUiState()
    data class Success(val devices: List<AcousticDevice>) : DevicesUiState()
    data class Error(val message: String) : DevicesUiState()
}

class DevicesViewModel(
    private val repository: DeviceRepository = DeviceRepository()
) : ViewModel() {

    private val _uiState = MutableStateFlow<DevicesUiState>(DevicesUiState.Loading)
    val uiState: StateFlow<DevicesUiState> = _uiState.asStateFlow()

    init {
        loadDevices()
    }

    private fun loadDevices() {
        viewModelScope.launch {
            repository.observeDevices().collect { result ->
                result.onSuccess { devices ->
                    _uiState.value = DevicesUiState.Success(devices)
                }.onFailure { exception ->
                    _uiState.value = DevicesUiState.Error(exception.message ?: "Unknown error")
                }
            }
        }
    }
}
