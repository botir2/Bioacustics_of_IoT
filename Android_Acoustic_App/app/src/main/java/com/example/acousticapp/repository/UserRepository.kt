package com.example.acousticapp.repository

import com.example.acousticapp.model.UserProfile
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.firestore.FieldValue
import com.google.firebase.firestore.FirebaseFirestore
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.tasks.await

class UserRepository(
    private val auth: FirebaseAuth = FirebaseAuth.getInstance(),
    private val db: FirebaseFirestore = FirebaseFirestore.getInstance()
) {

    fun getUserProfile(): Flow<Result<UserProfile>> = callbackFlow {
        val currentUser = auth.currentUser
        if (currentUser == null) {
            trySend(Result.failure(Exception("No user logged in")))
            close()
            return@callbackFlow
        }

        val docRef = db.collection("users").document(currentUser.uid)
        
        val listener = docRef.addSnapshotListener { snapshot, e ->
            if (e != null) {
                trySend(Result.failure(e))
                return@addSnapshotListener
            }

            if (snapshot != null && snapshot.exists()) {
                val profile = snapshot.toObject(UserProfile::class.java)
                if (profile != null) {
                    trySend(Result.success(profile))
                } else {
                    trySend(Result.failure(Exception("Failed to parse user profile")))
                }
            } else {
                // Document doesn't exist, create it from FirebaseAuth data
                val newUserProfile = UserProfile(
                    uid = currentUser.uid,
                    name = currentUser.displayName,
                    email = currentUser.email,
                    photoUrl = currentUser.photoUrl?.toString(),
                    provider = currentUser.providerData.getOrNull(1)?.providerId ?: "email",
                    createdAt = com.google.firebase.Timestamp.now(),
                    lastLoginAt = com.google.firebase.Timestamp.now()
                )
                
                db.collection("users").document(currentUser.uid).set(newUserProfile)
                    .addOnSuccessListener {
                        trySend(Result.success(newUserProfile))
                    }
                    .addOnFailureListener { err ->
                        trySend(Result.failure(err))
                    }
            }
        }

        awaitClose { listener.remove() }
    }
}
