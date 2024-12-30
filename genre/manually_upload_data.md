# first, create tag in a format `data-genre-yyyy-mm-dd` push it and create release from the tag  
# then, set token variable in console:
```shell
export GITHUB_TOKEN={token} 
```
# next step is to get target release id:
```shell
curl -L -H "Accept: application/vnd.github+json" -H "Authorization: Bearer ${GITHUB_TOKEN}" -H "X-GitHub-Api-Version: 2022-11-28" https://api.github.com/repos/dzharikhin/zmt-bot/releases
```
# last step is to upload data:
```shell
find . -name 'snippets.z*' -printf '%f\n' | sort | tail -n +1 | xargs -I {} curl -L -X POST -H "Accept: application/vnd.github+json" -H "Authorization: Bearer ${GITHUB_TOKEN}" -H "X-GitHub-Api-Version: 2022-11-28" -H "Content-Type: application/octet-stream" "https://uploads.github.com/repos/dzharikhin/zmt-bot/releases/{release_id}/assets?name={}" --data-binary "@{}"
```
> if you need to continue upload from specific file, adjust `tail` parameters
