package com.example.acousticapp.model

data class DeviceStatus(
    val deviceId: String = "",
    val acousticStatus: String = "",
    val connectionStatus: String = "",
    val batteryLevel: Int = 0,
    val deviceMode: String = "",
    val lastSeen: Long = 0L
)
