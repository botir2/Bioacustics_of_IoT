package com.example.acousticapp.map

import com.example.acousticapp.model.DeviceDetectionInfo

sealed class DeviceMapState {
    object Idle : DeviceMapState()
    object Loading : DeviceMapState()
    data class Success(val detections: List<DeviceDetectionInfo>) : DeviceMapState()
    data class Error(val message: String) : DeviceMapState()
}
