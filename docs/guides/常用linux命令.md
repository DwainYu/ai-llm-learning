# 常用 Linux 命令速查

## 文件与目录操作

- 查看当前目录：`pwd`
- 列出文件：`ls -l`
- 切换目录：`cd 目录名`
- 创建目录：`mkdir 目录名`
- 删除文件：`rm 文件名`
- 删除目录及内容：`rm -rf 目录名`
- 复制文件/目录：`cp 源 目标`，如 `cp a.txt b.txt`，`cp -r dir1 dir2`
- 移动/重命名：`mv 源 目标`

## 文件内容查看与编辑

- 查看文件内容：`cat 文件名`
- 分页查看：`less 文件名` 或 `more 文件名`
- 实时查看文件尾部：`tail -f 文件名`
- 编辑文件：`nano 文件名` 或 `vim 文件名`

## 权限与用户

- 查看权限：`ls -l`
- 修改权限：`chmod 755 文件名`
- 修改所有者：`chown 用户名 文件名`
- 切换用户：`su 用户名` 或 `sudo -i -u 用户名`
- 进入 root 用户：`sudo -i` 或 `su -`

## 系统与进程

- 查看内存：`free -h`
- 查看磁盘空间：`df -h`
- 查看CPU/内存/进程：`top` 或 `htop`
- 查看端口监听：`ss -tlnp` 或 `netstat -tlnp`
- 查看当前用户：`whoami`
- 查看当前时间：`date`

## 网络

- 查看IP地址：`ip addr` 或 `ifconfig`
- 测试连通性：`ping 域名或IP`
- 下载文件：`wget URL` 或 `curl -O URL`

## 软件包管理（Debian/Ubuntu）

- 更新软件源：`sudo apt update`
- 安装软件包：`sudo apt install 包名`
- 卸载软件包：`sudo apt remove 包名`
- 升级系统：`sudo apt upgrade`

## 其他

- 查看命令帮助：`命令 --help` 或 `man 命令`
- 清屏：`clear`
- 退出终端：`exit`

---
常用快捷键：  
- `Tab` 自动补全  
- `Ctrl+C` 终止当前命令  
- `Ctrl+L` 清屏  
- `Ctrl+Z` 挂起当前进程
