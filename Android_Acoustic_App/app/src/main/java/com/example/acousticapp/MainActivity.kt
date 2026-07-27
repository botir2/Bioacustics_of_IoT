package com.example.acousticapp

import android.content.Intent
import android.os.Bundle
import android.view.MenuItem
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.app.AppCompatDelegate
import androidx.core.view.GravityCompat
import androidx.navigation.NavController
import androidx.navigation.fragment.NavHostFragment
import androidx.navigation.ui.*
import coil.load
import coil.transform.CircleCropTransformation
import com.example.acousticapp.databinding.ActivityMainBinding
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.firestore.FirebaseFirestore

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var navController: NavController
    private lateinit var appBarConfiguration: AppBarConfiguration

    override fun onCreate(savedInstanceState: Bundle?) {
        // Theme is now applied in SplashActivity for early set, 
        // but we keep a check here for robustness if MainActivity is deep-linked or started directly
        val prefs = getSharedPreferences("app_prefs", MODE_PRIVATE)
        val isNightMode = prefs.getBoolean("night_mode_enabled", false)
        val targetMode = if (isNightMode) AppCompatDelegate.MODE_NIGHT_YES else AppCompatDelegate.MODE_NIGHT_NO
        
        if (AppCompatDelegate.getDefaultNightMode() != targetMode) {
            AppCompatDelegate.setDefaultNightMode(targetMode)
        }

        super.onCreate(savedInstanceState)

        // Check login
        if (FirebaseAuth.getInstance().currentUser == null) {
            // TODO: Navigate to LoginActivity if implemented separately, 
            // or handle login within this activity if it's the start destination
            // For now, assume MainActivity is only for logged in users
            // and redirection happened in SplashActivity or similar.
        }

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setSupportActionBar(binding.toolbar)

        val navHostFragment = supportFragmentManager
            .findFragmentById(R.id.nav_host_fragment) as NavHostFragment
        navController = navHostFragment.navController

        appBarConfiguration = AppBarConfiguration(
            setOf(R.id.nav_dashboard, R.id.nav_map, R.id.nav_devices, R.id.nav_history, R.id.nav_profile),
            binding.drawerLayout
        )

        setupActionBarWithNavController(navController, appBarConfiguration)
        binding.navView.setupWithNavController(navController)
        binding.bottomNav.setupWithNavController(navController)
        
        setupNavHeader()
        setupNightMode()
        
        binding.navView.setNavigationItemSelectedListener { menuItem ->
            if (menuItem.itemId == R.id.nav_logout) {
                // Handle logout from drawer
                FirebaseAuth.getInstance().signOut()

                // Clear Google Sign-In session
                val gso = com.google.android.gms.auth.api.signin.GoogleSignInOptions.Builder(com.google.android.gms.auth.api.signin.GoogleSignInOptions.DEFAULT_SIGN_IN)
                    .requestEmail()
                    .build()
                com.google.android.gms.auth.api.signin.GoogleSignIn.getClient(this, gso).signOut()

                val intent = Intent(this, LoginActivity::class.java)
                intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
                startActivity(intent)
                finish()
                true
            } else if (menuItem.itemId == R.id.nav_night_mode) {
                // Night mode toggle handled in setupNightMode, 
                // but we need to return true here to prevent default navigation behavior
                val actionView = menuItem.actionView
                val drawerSwitch = actionView?.findViewById<androidx.appcompat.widget.SwitchCompat>(R.id.drawer_switch)
                drawerSwitch?.let { it.isChecked = !it.isChecked }
                true
            } else {
                val handled = NavigationUI.onNavDestinationSelected(menuItem, navController)
                if (handled) {
                    binding.drawerLayout.closeDrawer(GravityCompat.START)
                }
                handled
            }
        }
    }

    private fun setupNightMode() {
        val prefs = getSharedPreferences("app_prefs", MODE_PRIVATE)
        val isNightMode = prefs.getBoolean("night_mode_enabled", false)

        val menu = binding.navView.menu
        val nightModeItem = menu.findItem(R.id.nav_night_mode)
        
        // Update Label and Icon based on current mode
        if (isNightMode) {
            nightModeItem.title = getString(R.string.light_mode)
            nightModeItem.setIcon(R.drawable.ic_sun)
        } else {
            nightModeItem.title = getString(R.string.night_mode)
            nightModeItem.setIcon(R.drawable.ic_moon)
        }

        val actionView = nightModeItem.actionView
        val drawerSwitch = actionView?.findViewById<androidx.appcompat.widget.SwitchCompat>(R.id.drawer_switch)

        drawerSwitch?.isChecked = isNightMode

        drawerSwitch?.setOnCheckedChangeListener { _, isChecked ->
            prefs.edit().putBoolean("night_mode_enabled", isChecked).apply()
            AppCompatDelegate.setDefaultNightMode(
                if (isChecked) AppCompatDelegate.MODE_NIGHT_YES else AppCompatDelegate.MODE_NIGHT_NO
            )
        }
    }

    private fun setupNavHeader() {
        val headerView = binding.navView.getHeaderView(0)
        val ivProfile = headerView.findViewById<ImageView>(R.id.imageView)
        val tvName = headerView.findViewById<TextView>(R.id.tv_header_name)
        val tvEmail = headerView.findViewById<TextView>(R.id.tv_header_email)

        val user = FirebaseAuth.getInstance().currentUser
        if (user != null) {
            // Set initial values from FirebaseAuth
            tvName.text = user.displayName ?: "User"
            tvEmail.text = user.email ?: "No email"
            if (user.photoUrl != null) {
                ivProfile.load(user.photoUrl) {
                    crossfade(true)
                    transformations(CircleCropTransformation())
                    placeholder(android.R.drawable.sym_def_app_icon)
                    error(android.R.drawable.sym_def_app_icon)
                }
            }

            // Real-time update from Firestore
            FirebaseFirestore.getInstance().collection("users").document(user.uid)
                .addSnapshotListener { snapshot, e ->
                    if (e != null || snapshot == null || !snapshot.exists()) return@addSnapshotListener
                    
                    val name = snapshot.getString("name") ?: user.displayName ?: "User"
                    val email = snapshot.getString("email") ?: user.email ?: "No email"
                    val photoUrl = snapshot.getString("photoUrl") ?: user.photoUrl?.toString()

                    tvName.text = name
                    tvEmail.text = email
                    
                    if (!photoUrl.isNullOrEmpty()) {
                        ivProfile.load(photoUrl) {
                            crossfade(true)
                            transformations(CircleCropTransformation())
                            placeholder(android.R.drawable.sym_def_app_icon)
                            error(android.R.drawable.sym_def_app_icon)
                        }
                    }
                }
        }
    }

    override fun onSupportNavigateUp(): Boolean {
        return navController.navigateUp(appBarConfiguration) || super.onSupportNavigateUp()
    }

    override fun onBackPressed() {
        if (binding.drawerLayout.isDrawerOpen(GravityCompat.START)) {
            binding.drawerLayout.closeDrawer(GravityCompat.START)
        } else {
            super.onBackPressed()
        }
    }
}
