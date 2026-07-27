package com.example.acousticapp.model

data class DeviceDetectionInfo(
    val detectedId: String = "",
    val deviceId: String = "",
    val uid: String = "",
    val detectedClass: String = "",
    val confidence: Double = 0.0,
    val detectedTime: Long = 0L,
    val acousticStatus: String = "",
    val connectionStatus: String = "",
    val batteryLevel: Int = 0,
    val deviceMode: String = "",
    val latitude: Double = 0.0,
    val longitude: Double = 0.0,
    val gpsAddress: String = ""
)
