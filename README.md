# 環境建立與套件安裝

## 1.執行環境
- 作業系統：Ubuntu
- Python 版本：Python 3.8
- 虛擬環境工具：venv
- 套件安裝工具：pip

## 2.方法與步驟

### 1) 確認系統是否已有 Python 3.8
首先確認 Ubuntu 系統是否已安裝 `python3.8`：

```bash
python3.8 --version
```

### 2) 安裝 Python 3.8（若系統找不到 python3.8）
若 python3.8 --version 顯示找不到指令，請先安裝：
```bash
sudo apt update
sudo apt install -y python3.8 python3.8-venv python3.8-dev
```
### 3) 建立虛擬環境並啟用
在專案資料夾內建立一個名為 venv38 的虛擬環境資料夾。
```bash
python3.8 -m venv venv38
source venv38/bin/activate
```

### 4) 升級套件並安裝requirements.txt
```bash
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
```