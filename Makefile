repl:
	ts-node

autoformat:
	npx prettier --write .

autoformat-watcher:
	npm run-script prettier-watch

install-prettier:
	npm install --save-dev --save-exact prettier
