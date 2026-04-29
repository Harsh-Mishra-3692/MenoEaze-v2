const { exec } = require('child_process');
const fs = require('fs');

exec('npx next build', (error, stdout, stderr) => {
    const output = `STDOUT:\n${stdout}\n\nSTDERR:\n${stderr}\n\nERROR:\n${error}`;
    fs.writeFileSync('build_output_check.txt', output);
    console.log('Build check logged.');
});
