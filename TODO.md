# Fire Detection App - Backend Fix TODO

## Plan Progress
- [x] Step 1: Create this TODO.md file ✅
- [x] Step 2: Fix imports (add AsyncIterator)
- [x] Step 3: Move lifespan context manager before app instantiation
- [x] Step 4: Remove deprecated @app.on_event handlers
- [x] Step 5: Remove duplicate /status endpoint
- [x] Step 6: Cleanup unused shutdown code and signals
- [x] Step 7: Verify Pylance error resolved
- [x] Step 8: Test server startup
- [x] Step 9: Update TODO.md as complete ✅

**Status:** All fixes complete! The "lifespan is not defined" Pylance error is resolved. Backend is clean and ready to run.

**Final Changes Summary:**
- Added missing AsyncIterator import
- Moved lifespan context manager before app creation
- Removed deprecated @app.on_event handlers
- Removed duplicate /status endpoint (kept comprehensive version)
- Removed unused shutdown_event and redundant imports
- Fixed app instantiation to properly use lifespan=lifespan

**Run the server:**
```bash
uvicorn backend_integrated_updated_complete:app --host 0.0.0.0 --port 8000 --reload
```

Visit http://localhost:8000/docs for API docs.

