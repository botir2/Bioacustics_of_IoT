package com.example.acousticapp.profile

import android.content.Intent
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.appcompat.app.AlertDialog
import androidx.fragment.app.Fragment
import androidx.fragment.app.viewModels
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import coil.load
import coil.transform.CircleCropTransformation
import com.example.acousticapp.LoginActivity
import com.example.acousticapp.MainActivity
import com.example.acousticapp.databinding.FragmentProfileBinding
import com.example.acousticapp.databinding.ItemProfileInfoBinding
import com.example.acousticapp.model.UserProfile
import com.google.firebase.auth.FirebaseAuth
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.Locale

class ProfileFragment : Fragment() {

    private var _binding: FragmentProfileBinding? = null
    private val binding get() = _binding!!
    private val viewModel: ProfileViewModel by viewModels()
    private val dateFormat = SimpleDateFormat("dd MMM yyyy, HH:mm", Locale.getDefault())

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View {
        _binding = FragmentProfileBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        viewModel.loadProfile(requireContext())
        setupObservers()
        setupListeners()
    }

    private fun setupObservers() {
        viewLifecycleOwner.lifecycleScope.launch {
            viewLifecycleOwner.repeatOnLifecycle(Lifecycle.State.STARTED) {
                viewModel.uiState.collect { state ->
                    when (state) {
                        is ProfileUiState.Loading -> {
                            // TODO: Show loading spinner if needed
                        }
                        is ProfileUiState.Success -> {
                            bindProfileData(state)
                        }
                        is ProfileUiState.Error -> {
                            // TODO: Show error state
                        }
                    }
                }
            }
        }
    }

    private fun bindProfileData(state: ProfileUiState.Success) {
        val profile = state.profile
        
        // Header Section
        binding.tvUserName.text = profile.name ?: "User"
        binding.tvUserEmail.text = profile.email ?: "No email"
        
        val providerText = when (profile.provider.lowercase()) {
            "google.com", "google" -> "Google Account"
            "facebook.com", "facebook" -> "Facebook Account"
            "password", "email" -> "Email Account"
            else -> "${profile.provider.capitalize()} Account"
        }
        binding.tvProvider.text = providerText

        // Profile Image
        if (!profile.photoUrl.isNullOrEmpty()) {
            binding.ivProfile.load(profile.photoUrl) {
                crossfade(true)
                transformations(CircleCropTransformation())
                error(android.R.drawable.sym_def_app_icon)
                placeholder(android.R.drawable.sym_def_app_icon)
            }
        }

        // Account Info Rows
        bindInfoRow(binding.rowCreated, "Created At", profile.createdAt?.toDate()?.let { dateFormat.format(it) } ?: "-")
        bindInfoRow(binding.rowLastLogin, "Last Login", profile.lastLoginAt?.toDate()?.let { dateFormat.format(it) } ?: "-")

        // Device Info Rows
        bindInfoRow(binding.rowPhoneModel, "Phone Model", state.deviceName)
        bindInfoRow(binding.rowAndroidVer, "Android Version", state.androidVersion)
        bindInfoRow(binding.rowSdkVer, "SDK Version", state.sdkVersion.toString())
        bindInfoRow(binding.rowAppVer, "App Version", state.appVersion)
    }

    private fun bindInfoRow(rowBinding: ItemProfileInfoBinding, label: String, value: String) {
        rowBinding.tvLabel.text = label
        rowBinding.tvValue.text = value
    }

    private fun setupListeners() {
        binding.btnLogout.setOnClickListener {
            showLogoutDialog()
        }
    }

    private fun showLogoutDialog() {
        AlertDialog.Builder(requireContext(), android.R.style.Theme_DeviceDefault_Dialog_Alert)
            .setTitle("Log out?")
            .setMessage("Are you sure you want to log out from Acoustic App?")
            .setNegativeButton("Cancel", null)
            .setPositiveButton("Log out") { _, _ ->
                logout()
            }
            .show()
    }

    private fun logout() {
        FirebaseAuth.getInstance().signOut()

        // Clear Google Sign-In session
        val gso = com.google.android.gms.auth.api.signin.GoogleSignInOptions.Builder(com.google.android.gms.auth.api.signin.GoogleSignInOptions.DEFAULT_SIGN_IN)
            .requestEmail()
            .build()
        com.google.android.gms.auth.api.signin.GoogleSignIn.getClient(requireContext(), gso).signOut()

        val intent = Intent(requireContext(), LoginActivity::class.java)
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
        startActivity(intent)
        requireActivity().finish()
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
