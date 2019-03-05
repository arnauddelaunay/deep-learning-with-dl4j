rootTestDir="/tmp/dl4j_w2vSentiment/aclImdb/test/"
sentiment=$1

rootDir=$rootTestDir$sentiment"/"

for file in $(ls $rootDir)
do
    countWord=$(wc -w $rootDir$file)
    IFS=' ' read -r count filePath <<< $(echo $countWord)
    if [ $count -lt 20 ]
    then
        cat $filePath
        echo ""
    fi
done
