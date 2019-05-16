\c share_ai

--Insert test users
BEGIN;
INSERT INTO account values ('Andriy', '$2a$10$R8FL847WLSKF0.3iLwiSxOernsUmuizm8QWQGY0dtuugAhfFDJBGu', current_timestamp, 'user'); --password andriypassword
INSERT INTO account values ('Fidel', '$2a$10$ef151tkpbvtu.Spt6BBTqOj09f7i8IaXGTRw7phNlCWJrZoXc1o22', current_timestamp, 'user'); --password fidelpassword
INSERT INTO account values ('test_user_1', '$2a$10$mVFHmjqpERPYh1FjWh9G/eKlXE6LGycmeMeFzwGGyNOAKwXEcarmW', current_timestamp, 'user'); --password test_user_1
INSERT INTO account values ('BobUser', '$2a$10$n3W8ekUIOgR0pClXgeEKMOX3eidoDISQPoTN69V0DaYCIdL5XBeBe', current_timestamp, 'user'); --password root
INSERT INTO account values ('BillUser', '$2a$10$n3W8ekUIOgR0pClXgeEKMOX3eidoDISQPoTN69V0DaYCIdL5XBeBe', current_timestamp, 'user'); --password root
INSERT INTO account values ('JoeUser', '$2a$10$n3W8ekUIOgR0pClXgeEKMOX3eidoDISQPoTN69V0DaYCIdL5XBeBe', current_timestamp, 'user'); --password root
INSERT INTO account values ('JohnUser', '$2a$10$n3W8ekUIOgR0pClXgeEKMOX3eidoDISQPoTN69V0DaYCIdL5XBeBe', current_timestamp, 'user'); --password root


INSERT INTO user_account values ('Andriy', 'Andriy', 'Mulyar', 'US', 'VA', 'Richmond', '1111 Cherry Lane', 'fill/me/with/a/real/image/url.png', 'andriy@gmail.com');
INSERT INTO user_account values ('Fidel', 'Fidel', 'Rodriguez', 'US', 'VA', 'Richmond', '1112 Cherry Lane', 'fill/me/with/a/real/image/url.png', 'fidel@gmail.com');
INSERT INTO user_account values ('test_user_1', 'Test', 'UserDisplay', 'US', 'VA', 'Richmond', '1113 Cherry Lane', 'fill/me/with/a/real/image/url.png', 'test@gmail.com');
INSERT INTO user_account values ('BobUser', 'Bob', 'Smith', 'US', 'VA', 'Richmond', '1113 Cherry Lane', 'fill/me/with/a/real/image/url.png', 'bob@gmail.com');
INSERT INTO user_account values ('BillUser', 'Bill', 'Smith', 'US', 'VA', 'Richmond', '1113 Cherry Lane', 'fill/me/with/a/real/image/url.png', 'bill@gmail.com');
INSERT INTO user_account values ('JohnUser', 'John', 'Smith', 'US', 'VA', 'Richmond', '1113 Cherry Lane', 'fill/me/with/a/real/image/url.png', 'john@gmail.com');
INSERT INTO user_account values ('JoeUser', 'Joe', 'Smith', 'US', 'VA', 'Richmond', '1113 Cherry Lane', 'fill/me/with/a/real/image/url.png', 'joe@gmail.com');

INSERT INTO organization values ('OpenAI - NLP', 'We have a ton of funding', 'OpenAI');
INSERT INTO organization values ('Andriy''s Organization', 'Models galore.', 'Virginia Commonwealth University');
INSERT INTO organization values ('Bob''s Organization', 'This belongs to Bob.', 'Virginia Commonwealth University');
INSERT INTO organization values ('Bill''s Organization', 'This belongs to Bill.', 'Virginia Commonwealth University');

INSERT INTO organization_affiliate values ('Fidel', 'OpenAI - NLP', 'owner');
INSERT INTO organization_affiliate values ('Andriy', 'Andriy''s Organization', 'owner');
INSERT INTO organization_affiliate values ('test_user_1', 'OpenAI - NLP', 'member');
INSERT INTO organization_affiliate values ('BobUser', 'Bob''s Organization', 'owner');
INSERT INTO organization_affiliate values ('BillUser', 'Bill''s Organization', 'owner');
INSERT INTO organization_affiliate values ('JoeUser', 'Bob''s Organization', 'member');
INSERT INTO organization_affiliate values ('JohnUser', 'Bill''s Organization', 'member');

INSERT INTO model values ('Mega Model', 'OpenAI - NLP', 'A description of the model', 'Python', 'Tensorflow', 'NLP','Information Extraction','Named Entity Recognition');
INSERT INTO model values ('Mega Model 2', 'OpenAI - NLP', 'A description of the model', 'Python', 'Tensorflow', 'NLP','Information Extraction','Named Entity Recognition');
INSERT INTO model values ('Mega Model 3', 'OpenAI - NLP', 'A description of the model', 'Python', 'Tensorflow', 'NLP','Information Extraction','Named Entity Recognition');
INSERT INTO model values ('Cano Model 1', 'OpenAI - NLP', 'A description of the model', 'Python', 'Tensorflow', 'NLP','Information Extraction','Named Entity Recognition');
INSERT INTO model values ('Cano Model 2', 'OpenAI - NLP', 'A description of the model', 'Python', 'Tensorflow', 'NLP','Information Extraction','Named Entity Recognition');
INSERT INTO model values ('Cano Model 3', 'OpenAI - NLP', 'A description of the model', 'Python', 'Tensorflow', 'NLP','Information Extraction','Named Entity Recognition');

INSERT INTO model values ('Cool Model 1', 'Bob''s Organization', 'A description of the model', 'Python', 'Tensorflow', 'NLP','Information Extraction','Named Entity Recognition');
INSERT INTO model values ('Cool Model 2', 'Bob''s Organization', 'A description of the model', 'Python', 'Tensorflow', 'NLP','Information Extraction','Named Entity Recognition');
INSERT INTO model values ('Cool Model 3', 'Bob''s Organization', 'A description of the model', 'Python', 'Tensorflow', 'NLP','Information Extraction','Named Entity Recognition');
INSERT INTO model values ('Good Model 1', 'Bill''s Organization', 'A description of the model', 'Python', 'Tensorflow', 'NLP','Information Extraction','Named Entity Recognition');
INSERT INTO model values ('Good Model 2', 'Bill''s Organization', 'A description of the model', 'Python', 'Tensorflow', 'NLP','Information Extraction','Named Entity Recognition');
INSERT INTO model values ('Great Model', 'Andriy''s Organization', 'A description of the model', 'Python', 'Tensorflow', 'NLP','Information Extraction','Named Entity Recognition');


INSERT INTO version values ('Mega Model', '0.0.1', 'First model version', 'url/model/is/stored/at.bin', 0.00, current_timestamp, '1.13.1');
INSERT INTO version values ('Mega Model', '0.0.2', 'Second model version', 'url/model/is/stored/at.bin', 0.00, current_timestamp, '1.13.1');
INSERT INTO version values ('Mega Model 2', '0.0.1', 'First model version', 'url/model/is/stored/at.bin', 0.00, current_timestamp, '1.14.1');
INSERT INTO version values ('Mega Model 3', '0.0.1', 'First model version', 'url/model/is/stored/at.bin', 0.00, current_timestamp, '1.13.1');
INSERT INTO version values ('Mega Model 3', '0.1.0', 'Second model version', 'url/model/is/stored/at.bin', 0.00, current_timestamp, '1.13.1');
INSERT INTO version values ('Mega Model 3', '0.1.1', 'Third model version', 'url/model/is/stored/at.bin', 0.00, current_timestamp, '1.13.1');
INSERT INTO version values ('Cano Model 1', '0.0.1', 'First model version', 'url/model/is/stored/at.bin', 0.00, current_timestamp, '1.12.1');
INSERT INTO version values ('Cano Model 2', '0.0.1', 'First model version', 'url/model/is/stored/at.bin', 0.00, current_timestamp, '1.13.1');
INSERT INTO version values ('Cano Model 3', '0.0.1', 'First model version', 'url/model/is/stored/at.bin', 0.00, current_timestamp, '1.13.1');
INSERT INTO version values ('Cano Model 3', '0.0.2', 'First model version', 'url/model/is/stored/at.bin', 0.00, current_timestamp, '1.13.1');

INSERT INTO version values ('Cool Model 1', '0.0.1', 'First model version', 'url/model/is/stored/at.bin', 0.00, current_timestamp, '1.12.1');
INSERT INTO version values ('Cool Model 2', '0.0.1', 'First model version', 'url/model/is/stored/at.bin', 0.00, current_timestamp, '1.13.1');
INSERT INTO version values ('Cool Model 3', '0.0.1', 'First model version', 'url/model/is/stored/at.bin', 0.00, current_timestamp, '1.13.1');
INSERT INTO version values ('Cool Model 3', '0.0.2', 'First model version', 'url/model/is/stored/at.bin', 0.00, current_timestamp, '1.13.1');

INSERT INTO version values ('Good Model 1', '0.0.1', 'First model version', 'url/model/is/stored/at.bin', 0.00, current_timestamp, '1.12.1');
INSERT INTO version values ('Good Model 2', '0.0.1', 'First model version', 'url/model/is/stored/at.bin', 0.00, current_timestamp, '1.13.1');
INSERT INTO version values ('Great Model', '0.0.1', 'First model version', 'url/model/is/stored/at.bin', 0.00, current_timestamp, '1.13.1');
INSERT INTO version values ('Great Model', '0.0.2', 'First model version', 'url/model/is/stored/at.bin', 0.00, current_timestamp, '1.13.1');

INSERT INTO invitation values ('Andriy''s Organization', 'test_user_1', 'Andriy', 'pending');
INSERT INTO invitation values ('Bob''s Organization', 'test_user_1', 'BobUser', 'pending');
INSERT INTO invitation values ('Bob''s Organization', 'Fidel', 'BobUser', 'pending');
INSERT INTO invitation values ('Bob''s Organization', 'Andriy', 'BobUser', 'pending');
INSERT INTO invitation values ('Bill''s Organization', 'Fidel', 'BillUser', 'pending');
INSERT INTO invitation values ('Bill''s Organization', 'Andriy', 'BillUser', 'pending');

COMMIT;





