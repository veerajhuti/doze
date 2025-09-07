FROM node:alpine

WORKDIR /app

COPY package.json package-lock.json ./

RUN apk update && apk upgrade && npm install --legacy-peer-deps

COPY . .

RUN npm run build

CMD ["npx", "serve", "-s", "build", "-l", "80"]