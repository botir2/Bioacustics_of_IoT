package com.example.acousticapp.auth

import android.content.Context
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

class AuthViewModel(private val repository: AuthRepository = AuthRepository()) : ViewModel() {

    private val _authState = MutableStateFlow<AuthState>(AuthState.Idle)
    val authState: StateFlow<AuthState> = _authState.asStateFlow()

    fun signInWithGoogleToken(idToken: String, context: Context) {
        viewModelScope.launch {
            _authState.value = AuthState.Loading
            repository.firebaseAuthWithGoogle(idToken, context) { result ->
                result.fold(
                    onSuccess = {
                        _authState.value = AuthState.Success
                    },
                    onFailure = { exception ->
                        _authState.value = AuthState.Error(exception.message ?: "Unknown error")
                    }
                )
            }
        }
    }

    fun signInWithFacebookToken(accessToken: String, context: Context) {
        viewModelScope.launch {
            _authState.value = AuthState.Loading
            repository.firebaseAuthWithFacebook(accessToken, context) { result ->
                result.fold(
                    onSuccess = {
                        _authState.value = AuthState.Success
                    },
                    onFailure = { exception ->
                        _authState.value = AuthState.Error(exception.message ?: "Unknown error")
                    }
                )
            }
        }
    }

    fun signInWithEmail(email: String, pass: String, context: Context) {
        viewModelScope.launch {
            _authState.value = AuthState.Loading
            repository.signInWithEmail(email, pass, context) { result ->
                result.fold(
                    onSuccess = {
                        _authState.value = AuthState.Success
                    },
                    onFailure = { exception ->
                        _authState.value = AuthState.Error(exception.message ?: "Unknown error")
                    }
                )
            }
        }
    }

    fun signUpWithEmail(email: String, pass: String, context: Context) {
        viewModelScope.launch {
            _authState.value = AuthState.Loading
            repository.signUpWithEmail(email, pass, context) { result ->
                result.fold(
                    onSuccess = {
                        _authState.value = AuthState.Success
                    },
                    onFailure = { exception ->
                        _authState.value = AuthState.Error(exception.message ?: "Unknown error")
                    }
                )
            }
        }
    }
}
