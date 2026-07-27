package com.example.acousticapp.map

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.acousticapp.model.AcousticDevice
import com.example.acousticapp.repository.MapRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

sealed class MapUiState {
    object Idle : MapUiState()
    object Loading : MapUiState()
    data class Success(val device: AcousticDevice) : MapUiState()
    data class Error(val message: String) : MapUiState()
}

class MapViewModel(private val repository: MapRepository = MapRepository()) : ViewModel() {

    private val _uiState = MutableStateFlow<MapUiState>(MapUiState.Idle)
    val uiState: StateFlow<MapUiState> = _uiState.asStateFlow()

    fun loadMapData() {
        viewModelScope.launch {
            _uiState.value = MapUiState.Loading
            repository.getDeviceLocation("device_001").fold(
                onSuccess = { device ->
                    _uiState.value = MapUiState.Success(device)
                },
                onFailure = { e ->
                    _uiState.value = MapUiState.Error(e.message ?: "Unknown error")
                }
            )
        }
    }
}
