package com.example.acousticapp.bluetooth

import android.Manifest
import android.app.Activity
import android.bluetooth.BluetoothAdapter
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.fragment.app.viewModels
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.example.acousticapp.databinding.FragmentBluetoothBinding
import com.example.acousticapp.databinding.ItemBluetoothDeviceBinding
import com.example.acousticapp.model.BluetoothConnectionState
import com.example.acousticapp.model.BluetoothDeviceUiModel
import com.example.acousticapp.repository.BluetoothRepository
import kotlinx.coroutines.launch

class BluetoothDevicesFragment : Fragment() {

    private var _binding: FragmentBluetoothBinding? = null
    private val binding get() = _binding!!

    private val viewModel: BluetoothDevicesViewModel by viewModels {
        object : ViewModelProvider.Factory {
            override fun <T : ViewModel> create(modelClass: Class<T>): T {
                return BluetoothDevicesViewModel(BluetoothRepository(requireContext())) as T
            }
        }
    }
    
    private lateinit var deviceAdapter: BluetoothDeviceAdapter

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val allGranted = permissions.entries.all { it.value }
        if (allGranted) {
            binding.cardPermission.visibility = View.GONE
            handleBluetoothEnableAndScan()
        } else {
            Toast.makeText(context, "Permissions required for BLE scanning", Toast.LENGTH_SHORT).show()
        }
    }

    private val enableBluetoothLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == Activity.RESULT_OK) {
            viewModel.checkBluetoothStatus()
            viewModel.startScan()
        } else {
            Toast.makeText(context, "Bluetooth must be enabled to scan nearby devices.", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View {
        _binding = FragmentBluetoothBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        setupRecyclerView()
        setupObservers()
        setupListeners()
        checkPermissionsOnInit()
    }

    override fun onResume() {
        super.onResume()
        viewModel.checkBluetoothStatus()
    }

    private fun checkPermissionsOnInit() {
        if (hasAllPermissions()) {
            binding.cardPermission.visibility = View.GONE
        } else {
            binding.cardPermission.visibility = View.VISIBLE
        }
    }

    private fun hasAllPermissions(): Boolean {
        return getRequiredPermissions().all {
            ContextCompat.checkSelfPermission(requireContext(), it) == PackageManager.PERMISSION_GRANTED
        }
    }

    private fun getRequiredPermissions(): Array<String> {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            arrayOf(Manifest.permission.BLUETOOTH_SCAN, Manifest.permission.BLUETOOTH_CONNECT)
        } else {
            arrayOf(Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.BLUETOOTH, Manifest.permission.BLUETOOTH_ADMIN)
        }
    }

    private fun setupRecyclerView() {
        deviceAdapter = BluetoothDeviceAdapter { device ->
            viewModel.connectToDevice(device)
        }
        binding.rvBluetoothDevices.layoutManager = LinearLayoutManager(context)
        binding.rvBluetoothDevices.adapter = deviceAdapter
    }

    private fun setupObservers() {
        viewLifecycleOwner.lifecycleScope.launch {
            viewLifecycleOwner.repeatOnLifecycle(Lifecycle.State.STARTED) {
                launch {
                    viewModel.uiState.collect { state ->
                        updateUi(state)
                    }
                }
                launch {
                    viewModel.isBluetoothEnabled.collect { isEnabled ->
                        updateBluetoothStatusCard(isEnabled)
                    }
                }
            }
        }
    }

    private fun updateBluetoothStatusCard(isEnabled: Boolean) {
        if (isEnabled) {
            binding.tvBluetoothStatus.text = "Enabled and ready"
            binding.tvBluetoothStatus.setTextColor(ContextCompat.getColor(requireContext(), android.R.color.holo_blue_dark))
        } else {
            binding.tvBluetoothStatus.text = "Disabled"
            binding.tvBluetoothStatus.setTextColor(ContextCompat.getColor(requireContext(), android.R.color.holo_red_dark))
        }
    }

    private fun updateUi(state: BluetoothUiState) {
        when (state) {
            is BluetoothUiState.Idle -> {
                binding.layoutEmpty.visibility = View.VISIBLE
                binding.layoutScanning.visibility = View.GONE
                binding.rvBluetoothDevices.visibility = View.GONE
            }
            is BluetoothUiState.Scanning -> {
                binding.layoutEmpty.visibility = View.GONE
                binding.layoutScanning.visibility = View.VISIBLE
                binding.rvBluetoothDevices.visibility = View.GONE
                binding.btnScan.text = "Scanning..."
            }
            is BluetoothUiState.Success -> {
                binding.layoutEmpty.visibility = if (state.devices.isEmpty()) View.VISIBLE else View.GONE
                binding.layoutScanning.visibility = View.GONE
                binding.rvBluetoothDevices.visibility = if (state.devices.isNotEmpty()) View.VISIBLE else View.GONE
                binding.btnScan.text = "Scan for Devices"
                deviceAdapter.submitList(state.devices)
            }
            is BluetoothUiState.Error -> {
                Toast.makeText(context, state.message, Toast.LENGTH_SHORT).show()
                binding.btnScan.text = "Scan for Devices"
            }
            else -> {}
        }
    }

    private fun setupListeners() {
        binding.btnScan.setOnClickListener {
            handleScanButtonClick()
        }
        
        binding.btnGrantPermission.setOnClickListener {
            requestPermissionLauncher.launch(getRequiredPermissions())
        }
    }

    private fun handleScanButtonClick() {
        if (hasAllPermissions()) {
            handleBluetoothEnableAndScan()
        } else {
            requestPermissionLauncher.launch(getRequiredPermissions())
        }
    }

    private fun handleBluetoothEnableAndScan() {
        if (viewModel.isBluetoothEnabled.value) {
            viewModel.startScan()
        } else {
            val enableBtIntent = Intent(BluetoothAdapter.ACTION_REQUEST_ENABLE)
            enableBluetoothLauncher.launch(enableBtIntent)
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }

    private class BluetoothDeviceAdapter(private val onConnectClick: (BluetoothDeviceUiModel) -> Unit) :
        RecyclerView.Adapter<BluetoothDeviceAdapter.ViewHolder>() {

        private var devices: List<BluetoothDeviceUiModel> = emptyList()

        fun submitList(newList: List<BluetoothDeviceUiModel>) {
            devices = newList.sortedByDescending { it.rssi }
            notifyDataSetChanged()
        }

        override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
            val binding = ItemBluetoothDeviceBinding.inflate(LayoutInflater.from(parent.context), parent, false)
            return ViewHolder(binding)
        }

        override fun onBindViewHolder(holder: ViewHolder, position: Int) {
            holder.bind(devices[position])
        }

        override fun getItemCount(): Int = devices.size

        inner class ViewHolder(private val binding: ItemBluetoothDeviceBinding) : RecyclerView.ViewHolder(binding.root) {
            fun bind(device: BluetoothDeviceUiModel) {
                binding.tvDeviceName.text = device.name
                binding.tvMacAddress.text = device.address
                binding.tvRssi.text = "${device.rssi} dBm"
                binding.tvConnectionStatus.text = device.connectionState.name

                when (device.connectionState) {
                    BluetoothConnectionState.CONNECTING -> {
                        binding.btnConnect.isEnabled = false
                        binding.btnConnect.text = "Connecting..."
                    }
                    BluetoothConnectionState.CONNECTED -> {
                        binding.btnConnect.isEnabled = false
                        binding.btnConnect.text = "Connected"
                        binding.tvConnectionStatus.setTextColor(binding.root.context.getColor(android.R.color.holo_green_dark))
                    }
                    else -> {
                        binding.btnConnect.isEnabled = true
                        binding.btnConnect.text = "Connect"
                        binding.tvConnectionStatus.setTextColor(binding.root.context.getColor(android.R.color.darker_gray))
                    }
                }

                binding.btnConnect.setOnClickListener {
                    onConnectClick(device)
                }
            }
        }
    }
}
