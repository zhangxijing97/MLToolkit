# MLToolkit

Create a new directory
```
mkdir repos
```

Move into folders
```
cd repos
```

Clone your repository from GitHub onto your computer
```
git clone git@github.com:zhangxijing97/MLToolkit.git
```

Test connected the repository you created on GitHub to your local machine or not
```
git remote -v
```

Create a new file
```
touch hello_world.txt
```

Get status
```
git status
```

Adds your hello_world.txt file to the staging area
```
git add hello_world.txt
```

Add message to the commit
```
git commit -m "Add hello_world.txt"
```

Get detail of author who made the commit and the date and time of when the commit was made, press “q” to escape
```
git log
```

Modify a file, open the directory in Visual Studio Code
```
code .
```

Push your work to GitHub
```
git push
```
or
```
git push origin main
```
