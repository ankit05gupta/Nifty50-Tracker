#!/usr/bin/env python3
"""
Nifty 50 Stock Tracker - Main CLI Entry Point

This script provides a command-line interface to run the Nifty 50 Stock Tracker.
It can launch the Streamlit dashboard, run tests, or set up the database.
"""

import sys
import subprocess
import argparse
from pathlib import Path

def main():
    """Main entry point for the Nifty 50 Stock Tracker application."""
    
    parser = argparse.ArgumentParser(
        description="Nifty 50 Stock Tracker - Technical Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py dashboard        # Launch Streamlit dashboard
  python main.py verify          # Run setup verification
  python main.py setup-db        # Initialize database
  python main.py --help          # Show this help message
        """
    )
    
    parser.add_argument(
        'command',
        choices=['dashboard', 'verify', 'setup-db', 'help'],
        help='Command to execute'
    )
    
    args = parser.parse_args()
    
    if args.command == 'dashboard':
        launch_dashboard()
    elif args.command == 'verify':
        run_verification()
    elif args.command == 'setup-db':
        setup_database()
    elif args.command == 'help':
        parser.print_help()
    else:
        parser.print_help()

def launch_dashboard():
    """Launch the Streamlit dashboard."""
    print("🚀 Launching Nifty 50 Stock Tracker Dashboard...")
    print("📊 Dashboard will open in your web browser")
    print("🔄 Press Ctrl+C to stop the application")
    print("-" * 50)
    
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 
            'frontend/streamlit_app.py'
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch dashboard: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")

def run_verification():
    """Run the setup verification script."""
    print("🔍 Running setup verification...")
    print("-" * 50)
    
    try:
        subprocess.run([sys.executable, 'setup_verification_test.py'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Verification failed: {e}")
        sys.exit(1)

def setup_database():
    """Initialize the SQLite database."""
    print("🗄️  Setting up SQLite database...")
    print("-" * 50)
    
    try:
        # Import and use the consolidated package
        from nifty_tracker.database import create_stock_table, fetch_and_store_all_stocks
        
        create_stock_table()
        print("✅ Database created successfully!")
        
        print("📊 Fetching and storing stock data...")
        fetch_and_store_all_stocks()
        print("✅ Database setup completed successfully!")
        
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Print welcome message
    print("=" * 60)
    print("📈 NIFTY 50 STOCK TRACKER")
    print("   Technical Analysis & Educational Tool")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path('frontend/streamlit_app.py').exists():
        print("❌ Error: Please run this script from the project root directory")
        print("   Make sure you're in the Nifty50-Tracker folder")
        sys.exit(1)
    
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        print("\n🎯 Quick Start:")
        print("   python main.py dashboard    # Launch the dashboard")
        print("   python main.py verify       # Verify your setup")
        print("   python main.py --help       # Show all options")
        print("\n")
        sys.exit(0)
    
    main()