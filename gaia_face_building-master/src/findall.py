import os
for root, dirs, files in os.walk("/dvmm-filer2/projects/Hearst/keyframes"):
    for file in files:
        if file.endswith(".png"):
             print(os.path.join(root, file))