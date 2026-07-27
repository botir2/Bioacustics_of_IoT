package com.example.acousticapp.auth

import android.content.Context
import android.os.Build
import android.provider.Settings
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.auth.FirebaseAuthInvalidCredentialsException
import com.google.firebase.auth.FirebaseAuthInvalidUserException
import com.google.firebase.auth.FirebaseAuthUserCollisionException
import com.google.firebase.auth.FirebaseAuthWeakPasswordException
import com.google.firebase.auth.FacebookAuthProvider
import com.google.firebase.auth.GoogleAuthProvider
import com.google.firebase.firestore.FieldValue
import com.google.firebase.firestore.FirebaseFirestore

class AuthRepository(
    private val auth: FirebaseAuth = FirebaseAuth.getInstance(),
    private val db: FirebaseFirestore = FirebaseFirestore.getInstance()
) {

    fun firebaseAuthWithGoogle(idToken: String, context: Context, onResult: (Result<Unit>) -> Unit) {
        val credential = GoogleAuthProvider.getCredential(idToken, null)
        auth.signInWithCredential(credential)
            .addOnCompleteListener { task ->
                if (task.isSuccessful) {
                    val user = auth.currentUser
                    if (user != null) {
                        saveUserDataAndDeviceData(user, "google", context, onResult)
                    } else {
                        onResult(Result.failure(Exception("User is null after successful sign in")))
                    }
                } else {
                    onResult(Result.failure(task.exception ?: Exception("Authentication failed")))
                }
            }
    }

    fun firebaseAuthWithFacebook(accessToken: String, context: Context, onResult: (Result<Unit>) -> Unit) {
        val credential = FacebookAuthProvider.getCredential(accessToken)
        auth.signInWithCredential(credential)
            .addOnCompleteListener { task ->
                if (task.isSuccessful) {
                    val user = auth.currentUser
                    if (user != null) {
                        saveUserDataAndDeviceData(user, "facebook", context, onResult)
                    } else {
                        onResult(Result.failure(Exception("User is null after successful sign in")))
                    }
                } else {
                    onResult(Result.failure(task.exception ?: Exception("Facebook authentication failed")))
                }
            }
    }

    fun signInWithEmail(email: String, pass: String, context: Context, onResult: (Result<Unit>) -> Unit) {
        auth.signInWithEmailAndPassword(email, pass)
            .addOnCompleteListener { task ->
                if (task.isSuccessful) {
                    val user = auth.currentUser
                    if (user != null) {
                        saveUserDataAndDeviceData(user, "email", context, onResult)
                    } else {
                        onResult(Result.failure(Exception("User is null after successful login")))
                    }
                } else {
                    val message = when (task.exception) {
                        is FirebaseAuthInvalidUserException -> "No account found with this email."
                        is FirebaseAuthInvalidCredentialsException -> "Wrong password."
                        else -> task.exception?.message ?: "Login failed"
                    }
                    onResult(Result.failure(Exception(message)))
                }
            }
    }

    fun signUpWithEmail(email: String, pass: String, context: Context, onResult: (Result<Unit>) -> Unit) {
        auth.createUserWithEmailAndPassword(email, pass)
            .addOnCompleteListener { task ->
                if (task.isSuccessful) {
                    val user = auth.currentUser
                    if (user != null) {
                        saveUserDataAndDeviceData(user, "email", context, onResult)
                    } else {
                        onResult(Result.failure(Exception("User is null after successful sign up")))
                    }
                } else {
                    val message = when (task.exception) {
                        is FirebaseAuthUserCollisionException -> "This email is already registered. Please sign in instead."
                        is FirebaseAuthWeakPasswordException -> "Password must be at least 6 characters."
                        is FirebaseAuthInvalidCredentialsException -> "Invalid email format."
                        else -> {
                            val msg = task.exception?.message ?: ""
                            if (msg.contains("ADMIN_ONLY_OPERATION", true) || msg.contains("CONFIGURATION_NOT_FOUND", true)) {
                                "Email/password login is not enabled in Firebase Console."
                            } else {
                                msg.ifEmpty { "Sign up failed" }
                            }
                        }
                    }
                    onResult(Result.failure(Exception(message)))
                }
            }
    }

    private fun saveUserDataAndDeviceData(
        user: com.google.firebase.auth.FirebaseUser,
        provider: String,
        context: Context,
        onResult: (Result<Unit>) -> Unit
    ) {
        // Fallback name from email if display name is null
        val fallbackName = user.email?.substringBefore("@")?.capitalize() ?: "User"
        
        val userMap = hashMapOf(
            "uid" to user.uid,
            "name" to (user.displayName ?: fallbackName),
            "email" to (user.email ?: "No email"),
            "photoUrl" to (user.photoUrl?.toString() ?: ""),
            "provider" to provider,
            "createdAt" to FieldValue.serverTimestamp(),
            "lastLoginAt" to FieldValue.serverTimestamp()
        )

        db.collection("users").document(user.uid)
            .set(userMap)
            .addOnSuccessListener {
                saveDeviceData(user.uid, context, onResult)
            }
            .addOnFailureListener { e ->
                onResult(Result.failure(e))
            }
    }

    private fun saveDeviceData(
        ownerUid: String,
        context: Context,
        onResult: (Result<Unit>) -> Unit
    ) {
        val androidId = Settings.Secure.getString(context.contentResolver, Settings.Secure.ANDROID_ID)
        val deviceMap = hashMapOf(
            "deviceId" to androidId,
            "ownerUid" to ownerUid,
            "deviceName" to Build.MODEL,
            "manufacturer" to Build.MANUFACTURER,
            "androidVersion" to Build.VERSION.RELEASE,
            "connectionStatus" to "online",
            "lastSeen" to FieldValue.serverTimestamp()
        )

        db.collection("devices").document(androidId)
            .set(deviceMap)
            .addOnSuccessListener {
                onResult(Result.success(Unit))
            }
            .addOnFailureListener { e ->
                onResult(Result.failure(e))
            }
    }
}
