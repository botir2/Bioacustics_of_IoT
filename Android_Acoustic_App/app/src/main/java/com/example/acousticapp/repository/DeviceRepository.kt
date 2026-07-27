package com.example.acousticapp.repository

import com.example.acousticapp.model.AcousticDevice
import com.google.firebase.firestore.FirebaseFirestore
import com.google.firebase.firestore.ListenerRegistration
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow

class DeviceRepository {
    private val firestore = FirebaseFirestore.getInstance()

    fun observeDevice(documentId: String): Flow<Result<AcousticDevice?>> = callbackFlow {
        val listener: ListenerRegistration = firestore.collection("devices").document(documentId)
            .addSnapshotListener { snapshot, error ->
                if (error != null) {
                    trySend(Result.failure(error))
                    return@addSnapshotListener
                }

                if (snapshot != null && snapshot.exists()) {
                    val device = snapshot.toObject(AcousticDevice::class.java)
                    trySend(Result.success(device))
                } else {
                    trySend(Result.success(null))
                }
            }

        awaitClose { listener.remove() }
    }

    fun observeDevices(): Flow<Result<List<AcousticDevice>>> = callbackFlow {
        val listener: ListenerRegistration = firestore.collection("devices")
            .addSnapshotListener { snapshot, error ->
                if (error != null) {
                    trySend(Result.failure(error))
                    return@addSnapshotListener
                }

                if (snapshot != null) {
                    val devices = snapshot.toObjects(AcousticDevice::class.java)
                    trySend(Result.success(devices))
                } else {
                    trySend(Result.success(emptyList()))
                }
            }

        awaitClose { listener.remove() }
    }
}
