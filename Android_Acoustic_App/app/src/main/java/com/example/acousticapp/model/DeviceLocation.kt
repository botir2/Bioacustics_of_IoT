package com.example.acousticapp.model

data class DeviceLocation(
    val id: String = "",
    val name: String = "",
    val location: String = "",
    val latitude: Double = 0.0,
    val longitude: Double = 0.0,
    val status: String = "",
    val batteryLevel: Int = 0
)
