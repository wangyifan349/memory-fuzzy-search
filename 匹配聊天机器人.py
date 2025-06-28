import json
import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox, ttk
import threading
import os
import hashlib

# LCS algorithm (non-nested, no list comprehension)
def lcs_length(str1, str2):
    """Compute the length of the longest common subsequence between two strings"""
    m = len(str1)
    n = len(str2)
    dp = []
    for i in range(m+1):
        dp_row = []
        for j in range(n+1):
            dp_row.append(0)
        dp.append(dp_row)

    for i in range(1, m+1):
        for j in range(1, n+1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                if dp[i-1][j] > dp[i][j-1]:
                    dp[i][j] = dp[i-1][j]
                else:
                    dp[i][j] = dp[i][j-1]
    return dp[m][n]

def similarity_score(query, question):
    """Calculate similarity score based on LCS"""
    lcs = lcs_length(query, question)
    avg_len = (len(query) + len(question)) / 2
    if avg_len == 0:
        return 0
    score = lcs / avg_len
    return score

class QASystem:
    def __init__(self):
        """Initialize the QA system with an empty QA list"""
        self.qa_list = []
        self.qa_hash_set = set()  # For de-duplication
        self.history = []

    def add_qa_list(self, qa_list):
        """Add a list of QA pairs to the system with de-duplication"""
        added_count = 0
        for qa in qa_list:
            # Create a unique hash for each QA pair
            qa_hash = hashlib.md5((qa['question'] + qa['answer']).encode('utf-8')).hexdigest()
            if qa_hash not in self.qa_hash_set:
                self.qa_list.append(qa)
                self.qa_hash_set.add(qa_hash)
                added_count += 1
        return added_count

    def find_best_match(self, user_question, threshold=0.3):
        """Find the best matching answer"""
        best_score = 0
        best_answer = "Sorry, I cannot answer your question at the moment."
        for qa in self.qa_list:
            score = similarity_score(user_question, qa['question'])
            if score > best_score:
                best_score = score
                best_answer = qa['answer']
        if best_score >= threshold:
            return best_answer
        else:
            return "Sorry, I cannot answer your question at the moment."

    def ask(self, user_question):
        """Process user question"""
        answer = self.find_best_match(user_question)
        self.history.append({'question': user_question, 'answer': answer})
        return answer

def load_qa_from_file(filepath):
    """Load QA list from JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            qa_list = json.load(f)
            return qa_list
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load QA data: {e}")
        return None

class ChatWindow:
    def __init__(self):
        """Initialize chat window"""
        self.qa_system = QASystem()
        self.create_main_window()
        self.create_widgets()
        self.update_qa_count()

    def create_main_window(self):
        """Create main window with improved interface"""
        self.window = tk.Tk()
        self.window.title("Intelligent QA System")
        self.window.geometry("700x600")
        self.window.configure(bg="#E8E8E8")
        self.window.resizable(False, False)

    def create_widgets(self):
        """Create GUI widgets"""
        # Style configuration
        style = ttk.Style()
        style.theme_use('default')
        style.configure('TButton', font=('Helvetica', 12))
        style.configure('TLabel', font=('Helvetica', 12), background='#E8E8E8')
        style.configure('TRadiobutton', font=('Helvetica', 12), background='#E8E8E8')

        # Menu bar
        menu_bar = tk.Menu(self.window)
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Load QA Data", command=self.load_qa_files)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.window.quit)
        menu_bar.add_cascade(label="File", menu=file_menu)
        self.window.config(menu=menu_bar)

        # Chat display area
        self.chat_display = scrolledtext.ScrolledText(self.window, wrap=tk.WORD, font=('Helvetica', 12))
        self.chat_display.place(x=10, y=10, width=680, height=450)
        self.chat_display.configure(state='disabled', bg="#FFFFFF")

        # QA count label
        self.qa_count_label = ttk.Label(self.window, text="QA Pairs Loaded: 0")
        self.qa_count_label.place(x=10, y=470)

        # User input box
        self.user_input = ttk.Entry(self.window, font=('Helvetica', 12))
        self.user_input.place(x=10, y=500, width=580, height=30)
        self.user_input.bind("<Return>", self.send_message)

        # Send button
        self.send_button = ttk.Button(self.window, text="Send", command=self.send_message)
        self.send_button.place(x=600, y=500, width=80, height=30)

        # Clear chat button
        self.clear_button = ttk.Button(self.window, text="Clear Chat", command=self.clear_chat)
        self.clear_button.place(x=600, y=540, width=80, height=30)

    def load_qa_files(self):
        """Load multiple QA data files in a separate thread"""
        threading.Thread(target=self._load_qa_files_thread).start()

    def _load_qa_files_thread(self):
        """Thread function to load QA data files"""
        file_paths = filedialog.askopenfilenames(title='Select QA Data Files', filetypes=[('JSON files', '*.json')])
        if file_paths:
            total_added = 0
            for file_path in file_paths:
                qa_list = load_qa_from_file(file_path)
                if qa_list is not None:
                    added_count = self.qa_system.add_qa_list(qa_list)
                    total_added += added_count
                    file_name = os.path.basename(file_path)
                    self.display_message("System", f"Loaded {added_count} new QA pairs from '{file_name}'")
            self.update_qa_count()
            if total_added > 0:
                self.display_message("System", f"Total {total_added} new QA pairs loaded. You can start asking questions.")
            else:
                self.display_message("System", "No new QA pairs were added.")
        else:
            self.display_message("System", "No files were selected.")

    def update_qa_count(self):
        """Update the QA count label"""
        count = len(self.qa_system.qa_list)
        self.qa_count_label.config(text=f"QA Pairs Loaded: {count}")

    def send_message(self, event=None):
        """Send user message"""
        if not self.qa_system.qa_list:
            messagebox.showwarning("Notice", "Please load QA data first.")
            return
        user_text = self.user_input.get().strip()
        if user_text == "":
            return
        # Display user message
        self.display_message("User", user_text)
        # Get response
        threading.Thread(target=self._get_bot_response, args=(user_text,)).start()
        # Clear input box
        self.user_input.delete(0, tk.END)

    def _get_bot_response(self, user_text):
        """Get bot response in a separate thread"""
        bot_response = self.qa_system.ask(user_text)
        # Display bot response
        self.display_message("Assistant", bot_response)

    def display_message(self, sender, message):
        """Display message in chat"""
        self.chat_display.configure(state='normal')
        self.chat_display.insert(tk.END, f"{sender}: {message}\n\n")
        self.chat_display.see(tk.END)
        self.chat_display.configure(state='disabled')

    def clear_chat(self):
        """Clear chat history"""
        self.chat_display.configure(state='normal')
        self.chat_display.delete('1.0', tk.END)
        self.chat_display.configure(state='disabled')

    def run(self):
        """Run the chat window"""
        self.window.mainloop()

if __name__ == '__main__':
    chat_window = ChatWindow()
    chat_window.run()
