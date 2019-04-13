import praw
f = open("content.txt", "w+")
reddit = praw.Reddit(client_id='2n-JkPtToluQ1A',
                     client_secret='2GsEl4UykIXidjZUTtnHFItoR1Y',
                     user_agent='cs410')
print(reddit.read_only)
for submission in reddit.subreddit('depression').hot(limit=None):
    cur = ""
    title = submission.title
    title = title.encode("utf-8")
    content = submission.selftext
    content = content.encode("utf-8")
    cur += title.strip('\n')
    cur += content.strip()
    f.write(cur + '\n')
    #print(content)
f.close()


with open("all.txt", "r") as f:
    lines = f.readlines()
with open("stripped.txt", "w") as f:
    for line in lines:
        if len(line) > 950:
            f.write(line)

uniqlines = set(open('stripped.txt').readlines())

bar = open('StrippedPosts.txt', 'w').writelines(set(uniqlines))

fn = open("StrippedPosts.txt", "r")
lines = fn.readlines()
avg = sum([len(line.strip('\n')) for line in lines]) / len(lines)
print(avg)
