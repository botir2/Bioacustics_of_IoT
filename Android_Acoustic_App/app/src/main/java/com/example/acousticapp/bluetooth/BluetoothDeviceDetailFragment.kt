package com.example.acousticapp.bluetooth

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.fragment.app.viewModels
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import com.example.acousticapp.databinding.FragmentBluetoothDetailBinding
import kotlinx.coroutines.launch

class BluetoothDeviceDetailFragment : Fragment() {

    private var _binding: FragmentBluetoothDetailBinding? = null
    private val binding get() = _binding!!
    private val viewModel: BluetoothDeviceDetailViewModel by viewModels()

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View {
        _binding = FragmentBluetoothDetailBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        setupObservers()
        setupListeners()
    }

    private fun setupObservers() {
        viewLifecycleOwner.lifecycleScope.launch {
            viewLifecycleOwner.repeatOnLifecycle(Lifecycle.State.STARTED) {
                viewModel.uiState.collect { state ->
                    binding.tvConnectionStatus.text = state.connectionStatus.name
                    state.device?.let {
                        binding.tvDeviceName.text = it.name
                        binding.tvDeviceAddress.text = it.address
                    }
                }
            }
        }
    }

    private fun setupListeners() {
        binding.btnConnect.setOnClickListener { viewModel.connect() }
        binding.btnDisconnect.setOnClickListener { viewModel.disconnect() }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
