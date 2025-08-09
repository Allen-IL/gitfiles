# git常用指令整理：


##  基础配置
1. **设置用户名**  
   `git config --global user.name "你的名字"`
2. **设置邮箱**  
   `git config --global user.email "你的邮箱"`
3. **查看所有配置**  
   `git config --list`

---

##  仓库操作
4. **初始化新仓库**  
   `git init`
5. **克隆远程仓库**  
   `git clone <远程仓库URL>`
6. **添加远程仓库**  
   `git remote add <名称> <URL>`  
   (例：`git remote add origin https://github.com/user/repo.git`)
7. **重命名远程仓库**  
   `git remote rename <旧名> <新名>`
8. **查看远程仓库**  
   `git remote -v`

---

##  文件操作
9. **添加所有修改到暂存区**  
   `git add .`
10. **添加指定文件**  
    `git add <文件名>`
11. **提交暂存区内容**  
    `git commit -m "提交说明"`
12. **查看当前状态**  
    `git status`

---

##  分支管理
13. **创建新分支**  
    `git branch <分支名>`
14. **切换分支**  
    `git checkout <分支名>`
15. **创建并切换分支**  
    `git checkout -b <新分支名>`
16. **查看所有分支**  
    `git branch -a`
17. **删除本地分支**  
    `git branch -d <分支名>`
18. **删除远程分支**  
    `git push <远程名> --delete <分支名>`

---

##  远程同步
19. **推送到远程仓库**  
    `git push <远程名> <分支名>`
20. **首次推送并建立关联**  
    `git push -u <远程名> <分支名>`
21. **拉取远程更新**  
    `git pull <远程名> <分支名>`
22. **获取远程更新（不自动合并）**  
    `git fetch <远程名>`

---

##  版本控制
23. **查看提交历史**  
    `git log`
24. **简洁版历史**  
    `git log --oneline --graph -n 10`
25. **撤销工作区修改**  
    `git restore <文件名>`
26. **撤销暂存区修改**  
    `git restore --staged <文件名>`
27. **修改最近一次提交**  
    `git commit --amend`

---

##  合并与变基
28. **合并分支**  
    `git merge <要合并的分支名>`
29. **变基操作**  
    `git rebase <目标分支>`
30. **解决冲突后继续合并**  
    `git merge --continue`
31. **终止合并**  
    `git merge --abort`

---

##  标签管理
32. **创建标签**  
    `git tag <标签名>`
33. **查看所有标签**  
    `git tag`
34. **推送标签到远程**  
    `git push <远程名> <标签名>`
35. **推送所有标签**  
    `git push <远程名> --tags`

---

##  储藏操作
36. **临时保存修改**  
    `git stash`
37. **查看储藏列表**  
    `git stash list`
38. **恢复最近储藏**  
    `git stash pop`
39. **清空所有储藏**  
    `git stash clear`

---

##  高级操作
40. **比较文件差异**  
    `git diff`
41. **重置到指定提交**  
    `git reset --hard <commit_id>`
42. **强制推送（慎用）**  
    `git push -f <远程名> <分支名>`
43. **查看文件修改历史**  
    `git blame <文件名>`

---

##  实用技巧
- **设置推送默认关联**  
  `git config --global push.autoSetupRemote true`
- **设置默认编辑器为VSCode**  
  `git config --global core.editor "code --wait"`
- **查看命令帮助**  
  `git help <命令>` 或 `git <命令> -h`
