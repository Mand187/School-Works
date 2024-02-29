package main

import (
	"bufio"
	"fmt"
	"log"
	"net"
	"strings"
)

type client struct {
	message chan<- string
	name    string
}

var (
	entering = make(chan client)
	leaving  = make(chan client)
	messages = make(chan string) // all incoming client messages
)

func main() {
	listener, err := net.Listen("tcp", "localhost:8000")
	if err != nil {
		log.Fatal(err)
	}

	go broadcaster()
	for {
		conn, err := listener.Accept()
		if err != nil {
			log.Print(err)
			continue
		}
		go handleConn(conn)
	}
}

func broadcaster() {
	clients := make(map[client]bool) // all connected clients
	for {
		select {
		case msg := <-messages:
			// Broadcast incoming message to all
			// clients' outgoing message channels.
			for cli := range clients {
				cli.message <- msg
			}

		case cli := <-entering:
			clients[cli] = true
			var connectedClients string
			for c := range clients {
				connectedClients += c.name + ", "
			}
			cli.message <- "\nCurrent Clients: " + strings.TrimRight(connectedClients, ", ") + "\n"

		case cli := <-leaving:
			delete(clients, cli)
			close(cli.message)
		}
	}
}

func handleConn(conn net.Conn) {
	ch := make(chan string) // outgoing client messages
	go clientWriter(conn, ch)

	ch <- "Please enter a name: "
	nameScanner := bufio.NewScanner(conn)
	nameScanner.Scan()
	name := nameScanner.Text()

	client := client{message: ch, name: name}

	ch <- "You are " + name
	messages <- name + " has arrived"
	entering <- client

	input := bufio.NewScanner(conn)
	for input.Scan() {
		messages <- name + ": " + input.Text()
	}

	leaving <- client
	messages <- name + " has left"
	conn.Close()
}

func clientWriter(conn net.Conn, ch <-chan string) {
	for msg := range ch {
		fmt.Fprintln(conn, msg) // NOTE: ignoring network errors
	}
}
