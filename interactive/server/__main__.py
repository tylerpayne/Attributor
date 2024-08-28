if __name__ == "__main__":
    from server.app import app
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
