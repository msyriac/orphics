# Guide to contributing

The recommended contribution model is as follows. You first do the following once:

1. Fork this repo on github.com to get your own copy
2. Clone your fork to your local system. If you already had a clone of this repo (and not your fork), you can simply update the remote URL of origin to point to your new fork https://help.github.com/articles/changing-a-remote-s-url/
3. Add the original repo (wherever you forked it from) as a remote called `upstream` to your local system https://help.github.com/articles/configuring-a-remote-for-a-fork/

You can then make changes to your own fork and push to its own master, and these will not affect the main version owned by `msyriac`. But `msyriac` would very much encourage two things:

a. That you keep up to date with `msyriac`'s version because he may be fixing bugs and adding neat features (or new bugs)
b. That you contribute to `msyriac`'s version with your own new neat features and bug fixes

## Getting in sync

To do (a), you should regularly get in sync with my version. You do this as follows:

```
git fetch upstream
git merge upstream/master
```

This just fetches changes to the main version and then merges those changes in to whatever branch you are sitting on (presumably master).

## Contributing through pull requests

When you contribute code, you should contribute it in neat chunks that are logically separated from any other changes you have made. For example, maybe you've been working on your `master` for a while with some changes to parameter files and idiosyncracies of your use case. You don't want to bring all that baggage with you in a pull request. So the recommended procedure is to do the following:

1. Get in sync with upstream as described above
2. Create a new branch *from upstream/master*, not from your local master
```
git checkout upstream/master
git checkout -b nice_name_for_my_feature_branch
```
3. Make changes in this new branch -- only the changes that are logically connected to this feature or bug fix you are contributing
4. Commit and push this to origin
```
git commit -a -m 'descriptive message of fix`
git push origin nice_name_for_my_feature_branch
```
5. Go to your fork on github.com, go to this new branch, and submit a pull request with a description of your changes

Some things to note:
- For step (3), what if you've already made these changes in a branch (say, master) and you just want to copy them over? This is a little tricky and I'd appreciate suggestions from people on how to handle this case. What I do is to use emacs to switch between git branches in two separate buffers and then copy-paste the relevant changes. See https://msyriac.github.io/docs/productivity/emacs.html 
- You may get feedback from me or others on your contribution in a code review. You should make the necessary changes to `nice_name_for_my_feature_branch` and push them. That will automatically update the Pull Request.


