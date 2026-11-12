"""查看 Wulabel 使用统计（数据来自客户端的检查更新请求）。

用法：python stats.py [天数，默认 14]
"""

import sqlite3
import sys
import time
from pathlib import Path

DB_PATH = Path(__file__).with_name("feedback.db")


def day_offset(n: int) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(time.time() + 8 * 3600 - n * 86400))


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    if not DB_PATH.exists():
        print("还没有数据")
        return
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        conn.execute("SELECT 1 FROM installs LIMIT 1")
    except sqlite3.OperationalError:
        print("还没有数据")
        return

    def one(sql: str, *args) -> int:
        return conn.execute(sql, args).fetchone()[0]

    today = day_offset(0)
    print(f"累计安装（出现过的设备）：{one('SELECT COUNT(*) FROM installs')}")
    print(f"今天在用：{one('SELECT COUNT(*) FROM active WHERE day=?', today)}")
    print(f"近 7 天在用：{one('SELECT COUNT(DISTINCT device) FROM active WHERE day>=?', day_offset(6))}")
    print(f"近 30 天在用：{one('SELECT COUNT(DISTINCT device) FROM active WHERE day>=?', day_offset(29))}")

    print(f"\n最近 {days} 天（日期 / 在用 / 新增）：")
    for i in range(days - 1, -1, -1):
        d = day_offset(i)
        act = one("SELECT COUNT(*) FROM active WHERE day=?", d)
        new = one("SELECT COUNT(*) FROM installs WHERE first_day=?", d)
        print(f"  {d}  {act:>5}  {new:>5}")

    print("\n近 30 天各版本：")
    for v, n in conn.execute(
        "SELECT version, COUNT(*) FROM installs WHERE last_day>=? GROUP BY version ORDER BY 2 DESC",
        (day_offset(29),),
    ):
        print(f"  v{v}  {n}")

    print("\n近 30 天各系统：")
    for o, n in conn.execute(
        "SELECT os, COUNT(*) FROM installs WHERE last_day>=? GROUP BY os ORDER BY 2 DESC",
        (day_offset(29),),
    ):
        print(f"  {o}  {n}")
    conn.close()


if __name__ == "__main__":
    main()
