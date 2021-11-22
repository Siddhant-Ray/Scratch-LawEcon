if [ "$1" == "new" ]
then 
    docker run -d docker/getting-started
    docker pull doccano/doccano
    docker container create --name doccano \
    -e "ADMIN_USERNAME=admin" \
    -e "ADMIN_EMAIL=admin@example.com" \
    -e "ADMIN_PASSWORD=password" \
    -p 8000:8000 doccano/doccano
fi

docker container start doccano 


