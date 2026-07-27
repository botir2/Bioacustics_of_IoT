package com.example.acousticapp.map

import com.example.acousticapp.model.DeviceDetectionInfo
import com.google.firebase.firestore.FirebaseFirestore

class DeviceMapRepository(
    private val db: FirebaseFirestore = FirebaseFirestore.getInstance()
) {

    /**
     * Future function to fetch device detection markers for a specific owner.
     * Currently reads from "detections" collection filtered by uid.
     */
    fun getDeviceDetectionMarkers(
        ownerUid: String,
        onResult: (Result<List<DeviceDetectionInfo>>) -> Unit
    ) {
        db.collection("detections")
            .whereEqualTo("uid", ownerUid)
            .get()
            .addOnSuccessListener { querySnapshot ->
                val detections = querySnapshot.toObjects(DeviceDetectionInfo::class.java)
                onResult(Result.success(detections))
            }
            .addOnFailureListener { e ->
                onResult(Result.failure(e))
            }
    }
    
    /**
     * Future function to fetch device status/location from "devices" collection.
     */
    fun getDevices(
        ownerUid: String,
        onResult: (Result<List<com.example.acousticapp.model.DeviceLocation>>) -> Unit
    ) {
        db.collection("devices")
            .whereEqualTo("ownerUid", ownerUid)
            .get()
            .addOnSuccessListener { querySnapshot ->
                val devices = querySnapshot.toObjects(com.example.acousticapp.model.DeviceLocation::class.java)
                onResult(Result.success(devices))
            }
            .addOnFailureListener { e ->
                onResult(Result.failure(e))
            }
    }
}
