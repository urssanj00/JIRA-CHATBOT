# main.py
from src.chatbot import JIRAChatbot


def main():
    # Initialize chatbot
    chatbot = JIRAChatbot('data/jira_dataset.csv')

    print("JIRA Chatbot: Hello! How can I help you? (Type 'quit' to exit)")

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() == 'quit':
                print("JIRA Chatbot: Goodbye!")
                break

            response = chatbot.get_response(user_input)
            print("JIRA Chatbot:", response)

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print("Please try again with a different query.")


if __name__ == "__main__":
    main()