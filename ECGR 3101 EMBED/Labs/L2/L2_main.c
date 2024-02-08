//*****************************************************************************
//
// ECGR 3101 LAB 2 VER 0.1
//
// Author: Matthew Anderson
// Author: Ali Altabtabae
// Date:   11/04/2023
// Due Date: 11/06/2023
//
// Description:
// Uses UART to transmit a string throuhg a UART to USB cable and display on a terminal
//
//*****************************************************************************
#include <stdint.h>
#include <stdbool.h>
#include "inc/hw_memmap.h"
#include "driverlib/fpu.h"
#include "driverlib/gpio.h"
#include "driverlib/pin_map.h"
#include "driverlib/sysctl.h"
#include "driverlib/uart.h"

void print(char *data) {
    while (*data != '\0') {
        UARTCharPut(UART5_BASE, *data++);
    }
}

void UARTConFig() {
       FPUEnable();
       FPULazyStackingEnable();

       SysCtlPeripheralEnable(SYSCTL_PERIPH_UART5);
       SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOE);

       GPIOPinConfigure(GPIO_PE4_U5RX);
       GPIOPinConfigure(GPIO_PE5_U5TX);
       GPIOPinTypeUART(GPIO_PORTE_BASE, GPIO_PIN_4 | GPIO_PIN_5);

       UARTConfigSetExpClk(UART5_BASE, SysCtlClockGet(), 9600,
                           (UART_CONFIG_WLEN_8 | UART_CONFIG_STOP_ONE | UART_CONFIG_PAR_NONE));
}

int main(void) {

   UARTConFig();

   print("Hello, our names are Ali Altabtabae and Matthew Anderson\r\n");

    // Wait for interrupts to happen
    while (1);

}
