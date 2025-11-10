## Instrutions for generating patchnotes

Use the following guidelines and templates while generating patchnotes between commits:

Use `git status --porcelain` to view all differences

*Parse each file for their differences, if theres major changes, additions, or deletions (anything larger than updating spacing, indentation, or imports), then write patch notes like this*

All patchnotes go in the git commit message, seperate into sections per file with general description of file change (ex "updated existingfile.py", "added newfile.py", "removed oldfile.py")

Under each section name (listed above), create a list of major changes (additions, changes, deletions)

Keep the patchnotes concise and clean, do not add any emojis please - 