# ============================================================================
# @Time :  
# @Author: Wufei
# @File: gui.py
# ============================================================================
# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter.filedialog import askopenfilename, askdirectory
from tkinter.messagebox import showinfo
from creditprice import *
import pandas as pd
import re

class MainUI(tk.Frame):
    # Application构造函数，master为窗口的父控件
    def __init__(self, master=None):
        # 初始化Application的Frame部分
        tk.Frame.__init__(self, master)
        self.grid()
        self.path = tk.StringVar()
        self.createWidgets()

    def selectPath(self):
        '''选择要转换成excel的xmind地址'''
        self.path_ = askopenfilename()
        self.path.set(self.path_)
        # 创建控件

    def createWidgets(self):
        '''生成gui界面'''
        # 创建一个标签，输出要显示的内容
        self.firstLabel = tk.Label(self, text="目标路径")
        # 设定使用grid布局
        self.firstLabel.grid(row=0, column=0)
        self.firstEntry = tk.Entry(self, textvariable=self.path)
        self.firstEntry.grid(row=0, column=1)
        # 创建一个按钮，用来触发answer方法
        self.clickButton = tk.Button(self, text="路径选择", command=self.selectPath)
        # 设定使用grid布局
        self.clickButton.grid(row=0, column=2)
        #创建输入参数ld
        self.secondLabel = tk.Label(self, text='LGD')
        self.secondLabel.grid(row=1,column=0)
        self.ld = tk.StringVar()
        self.secEntry = tk.Entry(self, textvariable=self.ld)
        self.secEntry.insert(0,'1')
        self.secEntry.grid(row=1, column=1)
        #创建输入参数d
        self.thirdLabel = tk.Label(self, text='d')
        self.thirdLabel.grid(row=2,column=0)
        self.d = tk.StringVar()
        self.thiEntry = tk.Entry(self, textvariable=self.d)
        self.thiEntry.insert(0,'0')
        self.thiEntry.grid(row=2, column=1)
        #创建输入参数r_limit
        self.forthLabel = tk.Label(self, text='r_limit')
        self.forthLabel.grid(row=3,column=0)
        self.rl = tk.StringVar()
        self.forEntry = tk.Entry(self, textvariable=self.rl)
        self.forEntry.insert(0, '0')
        self.forEntry.grid(row=3, column=1)
        #创建输入参数risk_free
        self.fifthLabel = tk.Label(self, text='risk_free')
        self.fifthLabel.grid(row=4,column=0)
        self.rf = tk.StringVar()
        self.fifEntry = tk.Entry(self, textvariable=self.rf)
        self.fifEntry.insert(0, '0.04')
        self.fifEntry.grid(row=4, column=1)
        #创建输入参数score_name
        self.sixthLabel = tk.Label(self, text='score_name')
        self.sixthLabel.grid(row=5,column=0)
        self.score = tk.StringVar()
        self.sixEntry = tk.Entry(self, textvariable=self.score)
        self.sixEntry.insert(0, 'score')
        self.sixEntry.grid(row=5, column=1)
        #创建输入参数利率名
        self.sevenLabel = tk.Label(self, text='interest')
        self.sevenLabel.grid(row=6,column=0)
        self.interest = tk.StringVar()
        self.sevEntry = tk.Entry(self, textvariable=self.interest)
        self.sevEntry.insert(0, 'r')
        self.sevEntry.grid(row=6, column=1)
        #创建输入参数接受标签
        self.eightLabel = tk.Label(self, text='accept_flag')
        self.eightLabel.grid(row=7,column=0)
        self.flag = tk.StringVar()
        self.eigEntry = tk.Entry(self, textvariable=self.flag)
        self.eigEntry.insert(0, 'accept')
        self.eigEntry.grid(row=7, column=1)
        #创建输入参数bad_flag
        self.ninthLabel = tk.Label(self, text='bad_flag')
        self.ninthLabel.grid(row=8,column=0)
        self.flagy = tk.StringVar()
        self.ninEntry = tk.Entry(self, textvariable=self.flagy)
        self.ninEntry.insert(0, 'flagy')
        self.ninEntry.grid(row=8, column=1)

        #创建结果查看器
        self.t = tk.Text()
        self.t.grid(row=12, column=0)
        # 创建一个提交按钮，用来触发提交方法,获取值
        # 设定使用grid布局
        self.clickButton = tk.Button(self, text="提交", command=self.getvalue, font=("Arial", 10))
        self.clickButton.grid(row=9, column=1)


    def getvalue(self):
        '''获取输入的值，并计算利率'''
        global way, ld, d, rl, rf, score, interest, flag, y
        way = self.path.get()
        ld = float(self.ld.get())
        d = float(self.d.get())
        rl = float(self.rl.get())
        rf = float(self.rf.get())
        score = self.score.get()
        interest = self.interest.get()
        flag = self.flag.get()
        y = self.flagy.get()
        self.regvalue = '.*\.csv$|.*\.xlsx$|.*\.xls$'
        self.calc_reg = re.match(self.regvalue, way)
        if self.calc_reg:
            self.cp = credit_price(ld=ld, d = d, rl = rl, rf = rf, score=score, interest=interest, flag=flag, y = y)
            # self.cp = credit_price(ld=0.5, d = 1, rl = 0.05, rf = 0.05, score='score', interest='r', flag='accept', y = 'flagy')
            try:
                data = pd.read_csv(way, encoding='utf-8')
            except:
                data = pd.read_excel(way)
            # print(self.cp.calc_r_table(data))
            result = self.cp.calc_r_table(data)
            result.to_csv('result.csv', encoding='utf-8')
            self.t.delete(0.0, tk.END)
            self.t.insert('end', result.to_string())

            return 1
        else:
            showinfo(title='提示', message='请选择正确的数据文件，谢谢！')
            return 0
# 创建一个MainUI对象
app = MainUI()
# 设置窗口标题
app.master.title('「风险定价」')
# 设置窗体大小
app.master.geometry('600x600')
# 主循环开始
app.mainloop()