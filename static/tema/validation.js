function validateForm() {
    var email = document.getElementById('exampleInputEmail').value;
    var password = document.getElementById('exampleInputPassword').value;

    // Definir el correo electrónico y la contraseña correctos
    var correctEmail = 'usuario@ejemplo.com';
    var correctPassword = 'contraseña123';

    if (email === correctEmail && password === correctPassword) {
        // Si el correo electrónico y la contraseña son correctos, el formulario se puede enviar
        return true;
    } else {
        // Si el correo electrónico y/o la contraseña son incorrectos, mostrar un mensaje de error y evitar que se envíe el formulario
        alert('Correo electrónico o contraseña incorrectos.');
        return false;
    }
}
