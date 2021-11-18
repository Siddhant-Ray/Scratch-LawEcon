docker run -d docker/getting-started

if [ "$1" == "new" ]
then 
    docker pull doccano/doccano
    docker container create --name doccano \
    -e "ADMIN_USERNAME=admin" \
    -e "ADMIN_EMAIL=admin@example.com" \
    -e "ADMIN_PASSWORD=password" \
    -p 8000:8000 doccano/doccano

    docker container start doccano
else
    docker container start docanno
fi 


