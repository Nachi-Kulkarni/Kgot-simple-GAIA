# CLI Chat Interface Usage Examples

## Getting Started

1. **Start the server** (in one terminal):
   ```bash
   npm start
   ```

2. **Start the CLI chat** (in another terminal):
   ```bash
   npm run chat
   ```

## Example Conversations

### Research and Analysis
```
💬 You: Help me research the latest developments in quantum computing

🤖 Processing your request...

📝 Response:
──────────────────────────────────────────────────
[AI response with quantum computing research]
──────────────────────────────────────────────────

📊 Metadata:
Session: cli-session-1703123456789
Processing time: 2.3s
System used: Alita-KGoT Enhanced
```

### Code Generation
```
💬 You: Generate a Python script for data visualization

🤖 Processing your request...

📝 Response:
──────────────────────────────────────────────────
[Generated Python code with explanations]
──────────────────────────────────────────────────
```

### Using Commands
```
💬 You: /help

📖 Help - Available Commands:
──────────────────────────────
/help    - Show this help message
/clear   - Clear the screen
/session - Show current session information
/exit    - Exit the chat application

💡 Examples:
"Help me research AI developments"
"Generate a Python script for data analysis"
"Explain quantum computing concepts"
──────────────────────────────
```

## Advanced Features

### Session Management
- Each CLI session gets a unique session ID
- Session information is displayed in responses
- Use `/session` to view current session details

### Error Handling
- Automatic server connectivity checks
- Helpful troubleshooting messages
- Graceful error recovery

### Environment Configuration
Set custom server URL:
```bash
ALITA_SERVER_URL=http://localhost:8080 npm run chat
```

## Tips for Best Experience

1. **Keep the server running** - The CLI connects to the main Alita-KGoT server
2. **Use descriptive queries** - More specific requests get better responses
3. **Try different commands** - Explore `/help` for all available options
4. **Check connectivity** - If errors occur, verify the server is running on the correct port

## Troubleshooting

### Common Issues

**"Server error: 500"**
- Check if the main server is running (`npm start`)
- Verify all dependencies are installed (`npm install`)

**"ECONNREFUSED"**
- Ensure the server is running on the correct port (default: 3000)
- Check firewall settings

**"Module not found"**
- Run `npm install` to install missing dependencies
- Verify Node.js version (>=18.0.0)