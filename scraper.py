import praw
flairs = ['flair_name:"PROVIDING SUPPORT"','flair_name:"PROVIDING ADVICE"','flair_name:"MOTIVATION"','flair_name:"INSPIRATION"']

reddit = praw.Reddit(client_id='2n-JkPtToluQ1A',
                     client_secret='2GsEl4UykIXidjZUTtnHFItoR1Y',
                     user_agent='cs410')
print(reddit.read_only)
s = [0,0,0,0]
f = open('content.txt', "wb")
for i in range(len(flairs)):
    print('Adding', flairs[i], 'to content.txt.')
    for submission in reddit.subreddit('depression_help').search(flairs[i], syntax='cloudsearch', limit=1000):
        cur = ""
        title = submission.title
        title = title.strip('\n')
        #title = title.encode("utf-8")
        content = submission.selftext.strip()
        #content = content.encode("utf-8")
        cur = title + content + '/n'
        #cur += content
        f.write(cur.encode("utf-8"))
        s[i]+=1
print('Documents per filter/flair:',s) 
f.close()
   
with open('content.txt', "r", encoding="utf8") as f:
    lines = f.readlines()
f.close()

with open('data/helpDocs.txt', "w", encoding="utf8") as f:
    for line in lines:
        if len(line) > 200:
            f.write(line)    
f.close()            
print('Stripping and filtering.')
uniqlines = set(open('data/helpDocs.txt', encoding="utf8").readlines())
bar = open('data/helpDocs.txt', 'w',encoding="utf8").writelines(set(uniqlines))

fn = open('data/helpDocs.txt', "r",encoding="utf8")
lines = fn.readlines()

avg = sum([len(line.strip('\n')) for line in lines]) / len(lines)
print("Average length of a post is", avg)
print("\nAll documents added to data/helpDocs.txt!")
    