#!/usr/bin/env node

const readline = require('readline');
const fetch = require('node-fetch');

// CLI Chat Interface for Alita-KGoT Enhanced System
class ChatCLI {
  constructor() {
    this.rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout
    });
    this.sessionId = `cli-session-${Date.now()}`;
    this.serverUrl = process.env.ALITA_SERVER_URL || 'http://localhost:8888';
  }

  async sendMessage(message) {
    try {
      console.log('\n🤖 Processing your request...');
      console.log('📤 Sending request to:', `${this.serverUrl}/agent/execute`);
      console.log('📦 Request body:', JSON.stringify({
        message: message,
        sessionId: this.sessionId
      }));

      const response = await fetch(`${this.serverUrl}/agent/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: message,
          sessionId: this.sessionId
        })
      });

      console.log('📥 Received response:', response.status, response.statusText);

      if (!response.ok) {
        throw new Error(`Server error: ${response.status} ${response.statusText}`);
      }

      let result;
      try {
        result = await response.json();
        console.log('📋 Parsed response:', JSON.stringify(result, null, 2));
      } catch (parseError) {
        console.error('❌ JSON parse error:', parseError.message);
        const rawBody = await response.text();
        console.log('📄 Raw response body:', rawBody);
        throw new Error('Failed to parse server response');
      }

      console.log('\n📝 Response:');
      console.log('─'.repeat(50));
      console.log(result.response || result.message || 'No response received');
      console.log('─'.repeat(50));

      if (result.metadata) {
        console.log('\n📊 Metadata:');
        console.log(`Session: ${result.metadata.sessionId || this.sessionId}`);
        console.log(`Processing time: ${result.metadata.processingTime || 'N/A'}`);
        console.log(`System used: ${result.metadata.systemUsed || 'Unknown'}`);
      }

    } catch (error) {
      console.error('\n❌ Error:', error.message);
      console.log('\n💡 Troubleshooting:');
      console.log('- Check if the server is running on', this.serverUrl);
      console.log('- Verify your network connection');
      console.log('- Try restarting the Alita-KGoT server');
    }
  }

  showWelcome() {
    console.clear();
    console.log('\n🚀 Alita-KGoT Enhanced Chat CLI');
    console.log('═'.repeat(40));
    console.log('Welcome to the interactive chat interface!');
    console.log(`Session ID: ${this.sessionId}`);
    console.log(`Server: ${this.serverUrl}`);
    console.log('\n📋 Commands:');
    console.log('  /help    - Show this help message');
    console.log('  /clear   - Clear the screen');
    console.log('  /session - Show current session info');
    console.log('  /exit    - Exit the chat');
    console.log('\n💬 Type your message and press Enter to chat!');
    console.log('═'.repeat(40));
  }

  showHelp() {
    console.log('\n📖 Help - Available Commands:');
    console.log('─'.repeat(30));
    console.log('/help    - Show this help message');
    console.log('/clear   - Clear the screen');
    console.log('/session - Show current session information');
    console.log('/exit    - Exit the chat application');
    console.log('\n💡 Examples:');
    console.log('"Help me research AI developments"');
    console.log('"Generate a Python script for data analysis"');
    console.log('"Explain quantum computing concepts"');
    console.log('─'.repeat(30));
  }

  showSession() {
    console.log('\n📋 Session Information:');
    console.log('─'.repeat(25));
    console.log(`Session ID: ${this.sessionId}`);
    console.log(`Server URL: ${this.serverUrl}`);
    console.log(`Started: ${new Date().toLocaleString()}`);
    console.log('─'.repeat(25));
  }

  async handleCommand(input) {
    const command = input.trim().toLowerCase();

    switch (command) {
      case '/help':
        this.showHelp();
        break;
      case '/clear':
        console.clear();
        this.showWelcome();
        break;
      case '/session':
        this.showSession();
        break;
      case '/exit':
        console.log('\n👋 Thanks for using Alita-KGoT Enhanced Chat!');
        console.log('Goodbye!');
        this.rl.close();
        process.exit(0);
        break;
      default:
        console.log(`\n❓ Unknown command: ${input}`);
        console.log('Type /help for available commands.');
    }
  }

  async start() {
    this.showWelcome();

    const askQuestion = () => {
      this.rl.question('\n💬 You: ', async (input) => {
        if (!input.trim()) {
          askQuestion();
          return;
        }

        if (input.startsWith('/')) {
          await this.handleCommand(input);
        } else {
          await this.sendMessage(input);
        }

        askQuestion();
      });
    };

    askQuestion();
  }
}

// Handle graceful shutdown
process.on('SIGINT', () => {
  console.log('\n\n👋 Chat session ended. Goodbye!');
  process.exit(0);
});

// Start the CLI if this file is run directly
if (require.main === module) {
  const chat = new ChatCLI();
  chat.start().catch(console.error);
}

module.exports = ChatCLI;