# bash to free any misallocated memory:
# as root:
sync; echo 1 > /proc/sys/vm/drop_caches

# --- filters: --- #
# bash to remove all duplicated lines from file
awk '!seen[$0]++' precursor > filtered.txt

# regex to capture all lines containing any of a set of words:
^.*([bB]alcony|[tT]errace|[bB]ox|[dD]eck).*$

# regex to match any non-ascii character:
[^\x00-\x7F]

# bash to delete everything from file matching PATTERN:
sed '/PATTERN/d' file.txt > file2.txt

# bash to delete all lines from file containing non-ascii characters:
sed -i -E '/[^\x00-\x7F]/d' file.txt

# bash script to execute a command on a collection of files:
###
#!/bin/bash
for filename in ./*.txt; do
#   for ((i=0; i<=3; i++)); do
    newFilename="filtered/$(basename $filename)"
    echo $newFilename
    # remove all repeated lines:
    awk '!seen[$0]++' $filename > $newFilename
    # remove all lines that do not end in a punctuation mark:
    sed -i -E "/[^\.^\?^\!]$/d" $newFilename > "filtered/$( basename $filename)2.txt"
#   done
done
###

# return all lines in file2 that aren't in file1
grep -v -f file1 file2

# bash to remove all traces of a sensitive file from commit history:
git filter-branch --force --index-filter \
"git rm --cached --ignore-unmatch PATH_TO_SENSITIVE_FILE" \
--prune-empty --tag-name-filter cat -- --all

git push --force --verbose --dry-run
git push --force

# convert all files in the current directory form utf8 to ascii
for filename in ./*.txt; do
    newFilename="converted/$(basename $filename)"
    echo $newFilename
    # do the conversion
    konwert utf8-ascii $filename -o $newFilename
done

# replace all unnecessary newlines with spaces
for filename in ./*.txt; do
    vim $filename -c ":%s/\\S\\zs\\n\\ze/ /g" -c ":w" -c ":q!"
done

for filename in ./*.txt; do
    # replace all double qoutes with single quotes
    sed -i "s/\"/'/g" $filename
done
