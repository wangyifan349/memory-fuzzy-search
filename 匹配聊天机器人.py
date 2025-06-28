import sys
import json
import threading
import hashlib
import os

from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
                             QLineEdit, QPushButton, QLabel, QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal, QObject

# ---------------------------------------------------------

def lcs_length(str1, str2):
    """计算两个字符串的最长公共子序列长度"""
    m = len(str1)
    n = len(str2)
    dp = []
    for i in range(m+1):
        dp_row = []
        for j in range(n+1):
            dp_row.append(0)
        dp.append(dp_row)
    # 动态规划计算LCS长度
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

# ---------------------------------------------------------

def similarity_score(query, question):
    """基于LCS计算相似度得分"""
    lcs = lcs_length(query, question)
    avg_len = (len(query) + len(question)) / 2
    if avg_len == 0:
        return 0
    score = lcs / avg_len
    return score

# ---------------------------------------------------------

class QASystem:
    """问答系统类"""
    def __init__(self):
        self.qa_list = []          # 存储QA对的列表
        self.qa_hash_set = set()   # 用于去重的哈希集合
        self.history = []          # 记录对话历史

    def add_qa_list(self, qa_list):
        """添加QA列表，并进行去重处理"""
        added_count = 0
        for qa in qa_list:
            question = qa['question'].strip().replace('\n', ' ')
            answer = qa['answer'].strip().replace('\n', ' ')
            # 生成QA对的唯一哈希值
            qa_hash = hashlib.md5((question + answer).encode('utf-8')).hexdigest()
            if qa_hash not in self.qa_hash_set:
                self.qa_list.append({'question': question, 'answer': answer})
                self.qa_hash_set.add(qa_hash)
                added_count += 1
        return added_count

    def find_best_match(self, user_question, threshold=0.3):
        """查找最佳匹配的答案"""
        best_score = 0
        best_answer = "抱歉，我暂时无法回答您的问题。"
        for qa in self.qa_list:
            score = similarity_score(user_question, qa['question'])
            if score > best_score:
                best_score = score
                best_answer = qa['answer']
        if best_score >= threshold:
            return best_answer
        else:
            return "抱歉，我暂时无法回答您的问题。"

    def ask(self, user_question):
        """处理用户提问并返回答案"""
        answer = self.find_best_match(user_question)
        self.history.append({'question': user_question, 'answer': answer})
        return answer

# ---------------------------------------------------------

def load_qa_from_file(filepath):
    """从JSON文件加载QA列表"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            qa_list = json.load(f)
            return qa_list
    except Exception as e:
        QMessageBox.critical(None, "错误", f"加载QA数据失败：{e}")
        return None

# ---------------------------------------------------------

class SignalBus(QObject):
    """用于线程间通信的信号总线"""
    display_message_signal = pyqtSignal(str, str)
    update_qa_count_signal = pyqtSignal(int)

# ---------------------------------------------------------

class ChatWindow(QWidget):
    """聊天窗口类"""
    def __init__(self):
        super().__init__()
        self.qa_system = QASystem()
        self.signal_bus = SignalBus()
        self.init_ui()
        self.connect_signals()
        self.update_qa_count()

    def init_ui(self):
        """初始化界面"""
        self.setWindowTitle("智能问答系统")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("background-color: #F0F0F0;")
        self.setFixedSize(800, 600)

        # 主布局
        main_layout = QVBoxLayout()

        # 菜单按钮布局
        menu_layout = QHBoxLayout()
        self.load_button = QPushButton("加载QA数据")
        self.load_button.setFixedHeight(40)
        self.load_button.setStyleSheet("font-size: 16px;")
        self.clear_button = QPushButton("清空聊天")
        self.clear_button.setFixedHeight(40)
        self.clear_button.setStyleSheet("font-size: 16px;")
        self.exit_button = QPushButton("退出")
        self.exit_button.setFixedHeight(40)
        self.exit_button.setStyleSheet("font-size: 16px;")
        menu_layout.addWidget(self.load_button)
        menu_layout.addWidget(self.clear_button)
        menu_layout.addWidget(self.exit_button)
        menu_layout.addStretch()

        # 聊天显示区域
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("font-size: 16px; background-color: white;")
        self.chat_display.setFixedHeight(420)

        # QA计数标签
        self.qa_count_label = QLabel("已加载QA对数：0")
        self.qa_count_label.setStyleSheet("font-size: 14px;")

        # 输入区域布局
        input_layout = QHBoxLayout()
        self.user_input = QLineEdit()
        self.user_input.setStyleSheet("font-size: 16px;")
        self.user_input.setPlaceholderText("请输入您的问题...")
        self.send_button = QPushButton("发送")
        self.send_button.setFixedWidth(80)
        self.send_button.setStyleSheet("font-size: 16px;")
        input_layout.addWidget(self.user_input)
        input_layout.addWidget(self.send_button)

        # 组装布局
        main_layout.addLayout(menu_layout)
        main_layout.addWidget(self.chat_display)
        main_layout.addWidget(self.qa_count_label)
        main_layout.addLayout(input_layout)
        self.setLayout(main_layout)

    def connect_signals(self):
        """连接信号和槽"""
        self.load_button.clicked.connect(self.load_qa_files)
        self.clear_button.clicked.connect(self.clear_chat)
        self.exit_button.clicked.connect(self.close)
        self.send_button.clicked.connect(self.send_message)
        self.user_input.returnPressed.connect(self.send_message)
        self.signal_bus.display_message_signal.connect(self.display_message)
        self.signal_bus.update_qa_count_signal.connect(self.update_qa_count_label)

    def load_qa_files(self):
        """加载QA数据文件（新线程）"""
        threading.Thread(target=self._load_qa_files_thread).start()

    def _load_qa_files_thread(self):
        """QA数据加载线程函数"""
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(self, "选择QA数据文件", "",
                                                "JSON文件 (*.json);;所有文件 (*)", options=options)
        if files:
            total_added = 0
            total_files = len(files)
            for file_path in files:
                qa_list = load_qa_from_file(file_path)
                if qa_list is not None:
                    added_count = self.qa_system.add_qa_list(qa_list)
                    total_added += added_count
                    file_name = os.path.basename(file_path)
                    self.signal_bus.display_message_signal.emit("系统",
                            f"从 '{file_name}' 加载了 {added_count} 个新的QA对。")
            self.signal_bus.update_qa_count_signal.emit(len(self.qa_system.qa_list))
            if total_added > 0:
                self.signal_bus.display_message_signal.emit("系统",
                        f"已从 {total_files} 个文件加载 {total_added} 个新的QA对。")
            else:
                self.signal_bus.display_message_signal.emit("系统", "没有添加新的QA对。")
        else:
            self.signal_bus.display_message_signal.emit("系统", "未选择任何文件。")

    def update_qa_count_label(self, count):
        """更新QA计数标签"""
        self.qa_count_label.setText(f"已加载QA对数：{count}")

    def update_qa_count(self):
        """初始更新QA计数标签"""
        count = len(self.qa_system.qa_list)
        self.qa_count_label.setText(f"已加载QA对数：{count}")

    def send_message(self):
        """发送用户消息"""
        if not self.qa_system.qa_list:
            QMessageBox.warning(self, "提示", "请先加载QA数据。")
            return
        user_text = self.user_input.text().strip()
        if user_text == "":
            return
        self.display_message("用户", user_text)
        threading.Thread(target=self._get_bot_response, args=(user_text,)).start()
        self.user_input.clear()

    def _get_bot_response(self, user_text):
        """获取机器人回复（新线程）"""
        bot_response = self.qa_system.ask(user_text)
        self.signal_bus.display_message_signal.emit("小助手", bot_response)

    def display_message(self, sender, message):
        """在聊天窗口显示消息"""
        self.chat_display.append(f"<b>{sender}：</b> {message}<br><br>")
        self.chat_display.moveCursor(self.chat_display.textCursor().End)

    def clear_chat(self):
        """清空聊天历史"""
        self.chat_display.clear()

    def closeEvent(self, event):
        """重写关闭事件，添加确认提示"""
        reply = QMessageBox.question(self, '退出', '您确定要退出吗？',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

# ---------------------------------------------------------

def main():
    """主函数，运行聊天应用程序"""
    app = QApplication(sys.argv)
    window = ChatWindow()
    window.show()
    sys.exit(app.exec_())

# ---------------------------------------------------------

if __name__ == '__main__':
    main()
