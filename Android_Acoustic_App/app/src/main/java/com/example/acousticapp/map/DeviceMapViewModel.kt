package com.example.acousticapp.map

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

class DeviceMapViewModel(
    private val repository: DeviceMapRepository = DeviceMapRepository()
) : ViewModel() {

    private val _mapState = MutableStateFlow<DeviceMapState>(DeviceMapState.Idle)
    val mapState: StateFlow<DeviceMapState> = _mapState.asStateFlow()

    fun loadDeviceMarkers(ownerUid: String) {
        viewModelScope.launch {
            _mapState.value = DeviceMapState.Loading
            repository.getDeviceDetectionMarkers(ownerUid) { result ->
                result.fold(
                    onSuccess = { detections ->
                        _mapState.value = DeviceMapState.Success(detections)
                    },
                    onFailure = { exception ->
                        _mapState.value = DeviceMapState.Error(exception.message ?: "Failed to load markers")
                    }
                )
            }
        }
    }
}
