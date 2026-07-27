package com.example.acousticapp.model

import com.google.firebase.Timestamp

data class UserProfile(
    val uid: String = "",
    val name: String? = null,
    val email: String? = null,
    val photoUrl: String? = null,
    val provider: String = "",
    val createdAt: Timestamp? = null,
    val lastLoginAt: Timestamp? = null
)
