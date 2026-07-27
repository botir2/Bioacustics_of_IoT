package com.example.acousticapp.profile

import android.content.Context
import android.os.Build
import android.provider.Settings
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.acousticapp.model.UserProfile
import com.example.acousticapp.repository.UserRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

sealed class ProfileUiState {
    object Loading : ProfileUiState()
    data class Success(
        val profile: UserProfile,
        val deviceName: String,
        val androidVersion: String,
        val sdkVersion: Int,
        val appVersion: String,
        val androidId: String
    ) : ProfileUiState()
    data class Error(val message: String) : ProfileUiState()
}

class ProfileViewModel(
    private val repository: UserRepository = UserRepository()
) : ViewModel() {

    private val _uiState = MutableStateFlow<ProfileUiState>(ProfileUiState.Loading)
    val uiState: StateFlow<ProfileUiState> = _uiState.asStateFlow()

    fun loadProfile(context: Context) {
        viewModelScope.launch {
            repository.getUserProfile().collect { result ->
                result.onSuccess { profile ->
                    _uiState.value = ProfileUiState.Success(
                        profile = profile,
                        deviceName = "${Build.MANUFACTURER} ${Build.MODEL}",
                        androidVersion = Build.VERSION.RELEASE,
                        sdkVersion = Build.VERSION.SDK_INT,
                        appVersion = getAppVersion(context),
                        androidId = Settings.Secure.getString(context.contentResolver, Settings.Secure.ANDROID_ID)
                    )
                }.onFailure { e ->
                    _uiState.value = ProfileUiState.Error(e.message ?: "Unknown error")
                }
            }
        }
    }

    private fun getAppVersion(context: Context): String {
        return try {
            val pInfo = context.packageManager.getPackageInfo(context.packageName, 0)
            pInfo.versionName ?: "1.0"
        } catch (e: Exception) {
            "1.0"
        }
    }
}
