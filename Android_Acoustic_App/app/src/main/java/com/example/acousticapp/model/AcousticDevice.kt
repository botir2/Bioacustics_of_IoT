package com.example.acousticapp.model

import com.google.firebase.Timestamp
import com.google.firebase.firestore.GeoPoint

data class AcousticDevice(
    val deviceId: String = "",
    val deviceName: String = "",
    val deviceStatus: String = "",
    val connectionStatus: String = "",
    val batteryStatus: Int = 0,
    val mode: String = "",
    val geoLocation: GeoPoint? = null,
    val lastDetectionId: String = "",
    val lastDetectionLabel: String = "",
    val lastDetectionTime: Timestamp? = null,
    val lastSeen: Timestamp? = null,
    val ownerUid: String = ""
)
