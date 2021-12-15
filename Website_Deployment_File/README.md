# Step for deploying the application on cloud and hosting website

Download and install the Heroku CLI at https://devcenter.heroku.com/articles/heroku-cli

$ heroku login

Use Git to clone drug-recommend's source code to your local machine.

$ heroku git:clone -a drug-recommend
$ cd drug-recommend

Copy all the 7 files from the folder Website_Deployment_File to current directory which is "$ drug-recommend"

Make some changes to the code you just cloned and deploy them to Heroku using Git.

$ git add .
$ git commit -am "your comment"
$ git push heroku main

The process will take some time to upload and deploy on cloud. 

Once it is deployed the website will be accessible at 
https://drug-recommend.herokuapp.com/
